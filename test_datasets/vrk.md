https://github.com/slivka83/ru_rag_test_dataset/tree/main  Ru RAG Test Dataset

Датасет для тестирования русскоязычных RAG-систем.

Содержит следующие данные:

    Файлы (в папка files) - спарсенные страницы русской Википедии. Название файлов это id страницы. Открыть их можно по такому URL: https://ru.wikipedia.org/?curid=
    Датафрейм Pandas (сохраненный в формате Pickle - ru_rag_test_dataset.pkl) со следующими колонками:
        Вопрос
        Правильный ответ
        Контекст - параграф, в котором содержится правильный ответ
        Название файла, в котором содержится правильный ответ
    В ноутбуке RAG dataset.ipynb код для воспроизведения.

Датасет собран на основе датасета RuBQ 2.0 (https://github.com/vladislavneon/RuBQ). Из него отбирались только те
вопросы, ответ на которые содержатся только в одной статье, в одном параграфе.

___
https://github.com/vladislavneon/RuBQ/tree/master/RuBQ_2.0
RuBQ 2.0

In December 2020 we built the second version of RuBQ. The dataset extension is based on questions obtained through
search engine query suggestion services. The dataset doubled in size: RuBQ 2.0 contains 2,910 questions along with the
answers and SPARQL queries. We also expanded the dataset with machine reading comprehension capabilities: RuBQ 2.0
incorporates answer-bearing paragraphs from Wikipedia for the majority of questions. Thus, the dataset is now not only
suitable for the evaluation of KBQA, but also can be used to evaluate machine reading comprehension, paragraph
retrieval, and end-to-end open-domain question answering. The dataset can be also used for experiments in hybrid QA,
where KBQA and text-based QA can enrich and complement each other.

This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.


___
