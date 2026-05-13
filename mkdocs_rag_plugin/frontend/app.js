/**
 * MkDocs RAG Frontend Application
 *
 * Обработка запросов к RAG API с поддержкой SSE streaming
 */

// Конфигурация API
const API_BASE_URL = 'http://localhost:8000';
const QUERY_ENDPOINT = `${API_BASE_URL}/api/v1/query`;

// DOM элементы
const elements = {
    form: document.getElementById('query-form'),
    question: document.getElementById('question'),
    retrievalMode: document.getElementById('retrieval-mode'),
    topK: document.getElementById('top-k'),
    streamToggle: document.getElementById('stream-toggle'),
    askBtn: document.getElementById('ask-btn'),
    btnText: document.querySelector('.btn-text'),
    btnLoading: document.querySelector('.btn-loading'),
    responseSection: document.getElementById('response-section'),
    answerContent: document.getElementById('answer-content'),
    copyAnswerBtn: document.getElementById('copy-answer-btn'),
    metadataSection: document.getElementById('metadata-section'),
    retrievalTime: document.getElementById('retrieval-time'),
    generationTime: document.getElementById('generation-time'),
    modelName: document.getElementById('model-name'),
    sourcesSection: document.getElementById('sources-section'),
    sourcesCount: document.getElementById('sources-count'),
    sourcesList: document.getElementById('sources-list'),
    errorMessage: document.getElementById('error-message'),
    errorText: document.getElementById('error-text')
};

let currentAnswer = '';
let abortController = null;

/**
 * Инициализация приложения
 */
function init() {
    elements.form.addEventListener('submit', handleSubmit);
    elements.copyAnswerBtn.addEventListener('click', copyAnswer);
}

/**
 * Обработка отправки формы
 */
async function handleSubmit(event) {
    event.preventDefault();

    const question = elements.question.value.trim();
    if (!question) {
        showError('Пожалуйста, введите вопрос');
        return;
    }

    const topK = parseInt(elements.topK.value, 10) || 5;
    const mode = elements.retrievalMode.value;
    const stream = elements.streamToggle.checked;

    // Сброс предыдущего состояния
    hideError();
    hideResponse();
    currentAnswer = '';

    // Отмена предыдущего запроса если есть
    if (abortController) {
        abortController.abort();
    }
    abortController = new AbortController();

    setLoading(true);

    try {
        if (stream) {
            await sendStreamingRequest(question, topK, mode);
        } else {
            await sendRegularRequest(question, topK, mode);
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('Запрос отменен');
            return;
        }
        showError(error.message || 'Произошла ошибка при обработке запроса');
    } finally {
        setLoading(false);
    }
}

/**
 * Отправка обычного (не потокового) запроса
 */
async function sendRegularRequest(question, topK, mode) {
    const requestBody = {
        question: question,
        top_k: topK,
        mode: mode,
        stream: false
    };

    const response = await fetch(QUERY_ENDPOINT, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody),
        signal: abortController.signal
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Ошибка сервера: ${response.status}`);
    }

    const data = await response.json();
    displayResponse(data);
}

/**
 * Отправка потокового запроса с SSE
 */
async function sendStreamingRequest(question, topK, mode) {
    const requestBody = {
        question: question,
        top_k: topK,
        mode: mode,
        stream: true
    };

    const response = await fetch(QUERY_ENDPOINT, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody),
        signal: abortController.signal
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Ошибка сервера: ${response.status}`);
    }

    showResponse();

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let finalData = null;

    while (true) {
        const {done, value} = await reader.read();

        if (done) {
            break;
        }

        const chunk = decoder.decode(value, {stream: true});
        buffer += chunk;

        // Парсим SSE формат
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Оставляем неполную строку в буфере

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const dataStr = line.slice(6);

                if (dataStr === '[DONE]') {
                    continue;
                }

                try {
                    const data = JSON.parse(dataStr);

                    // Обновляем ответ по мере поступления токенов
                    if (data.answer) {
                        currentAnswer = data.answer;
                        updateAnswerDisplay(currentAnswer);
                    }

                    // Финальные данные
                    if (data.sources || data.metadata) {
                        finalData = data;
                    }
                } catch (e) {
                    console.warn('Ошибка парсинга JSON:', e);
                }
            }
        }
    }

    // Отображаем финальные данные
    if (finalData) {
        displayMetadata(finalData.metadata);
        displaySources(finalData.sources || []);
    }
}

/**
 * Обновление отображения ответа
 */
function updateAnswerDisplay(text) {
    elements.answerContent.textContent = text;
    // Прокрутка к концу ответа
    elements.answerContent.scrollTop = elements.answerContent.scrollHeight;
}

/**
 * Отображение полного ответа
 */
function displayResponse(data) {
    showResponse();

    currentAnswer = data.answer || '';
    elements.answerContent.textContent = currentAnswer;

    displayMetadata(data.metadata);
    displaySources(data.sources || []);
}

/**
 * Отображение метаданных
 */
function displayMetadata(metadata) {
    if (!metadata) return;

    elements.retrievalTime.textContent = metadata.retrieval_time_ms || '-';
    elements.generationTime.textContent = metadata.generation_time_ms || '-';
    elements.modelName.textContent = metadata.model || '-';
}

/**
 * Отображение источников
 */
function displaySources(sources) {
    elements.sourcesCount.textContent = sources.length;
    elements.sourcesList.innerHTML = '';

    if (sources.length === 0) {
        elements.sourcesSection.style.display = 'none';
        return;
    }

    elements.sourcesSection.style.display = 'block';

    sources.forEach((source, index) => {
        const card = document.createElement('div');
        card.className = 'source-card';

        const scorePercent = Math.round((source.score || 0) * 100);

        card.innerHTML = `
            <div class="source-title">${escapeHtml(source.title || 'Без названия')}</div>
            <a href="${escapeHtml(source.url || '#')}" target="_blank" class="source-url">
                ${escapeHtml(source.url || '')}
            </a>
            <div class="source-snippet">${escapeHtml(source.snippet || '')}</div>
            <span class="source-score">Релевантность: ${scorePercent}%</span>
        `;

        elements.sourcesList.appendChild(card);
    });
}

/**
 * Экранирование HTML
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Копирование ответа в буфер обмена
 */
async function copyAnswer() {
    if (!currentAnswer) {
        showError('Нечего копировать');
        return;
    }

    try {
        await navigator.clipboard.writeText(currentAnswer);

        // Временное изменение текста кнопки
        const originalText = elements.copyAnswerBtn.textContent;
        elements.copyAnswerBtn.textContent = '✅ Скопировано!';

        setTimeout(() => {
            elements.copyAnswerBtn.textContent = originalText;
        }, 2000);
    } catch (error) {
        showError('Не удалось скопировать текст');
    }
}

/**
 * Показ секции ответа
 */
function showResponse() {
    elements.responseSection.style.display = 'block';
    elements.sourcesSection.style.display = 'none';
    elements.sourcesList.innerHTML = '';
}

/**
 * Скрытие секции ответа
 */
function hideResponse() {
    elements.responseSection.style.display = 'none';
}

/**
 * Установка состояния загрузки
 */
function setLoading(loading) {
    if (loading) {
        elements.askBtn.disabled = true;
        elements.btnText.style.display = 'none';
        elements.btnLoading.style.display = 'inline';
        elements.form.classList.add('loading');
    } else {
        elements.askBtn.disabled = false;
        elements.btnText.style.display = 'inline';
        elements.btnLoading.style.display = 'none';
        elements.form.classList.remove('loading');
    }
}

/**
 * Показ сообщения об ошибке
 */
function showError(message) {
    elements.errorText.textContent = message;
    elements.errorMessage.style.display = 'block';

    // Автоскрытие через 10 секунд
    setTimeout(() => {
        hideError();
    }, 10000);
}

/**
 * Скрытие сообщения об ошибке
 */
function hideError() {
    elements.errorMessage.style.display = 'none';
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', init);
