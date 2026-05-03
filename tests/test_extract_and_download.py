# tests/test_extract_and_download.py


# class TestExtractAndDownload:
#     """Тесты для функции extract_and_download"""
#
#     def test_env_variable_is_set(self):
#         """Проверяет, что переменная окружения test установлена в true"""
#         assert os.getenv("test") == "true"
#
#     @patch("src.fetch_datasets.urllib.request.urlretrieve")
#     def test_download_with_settings(self, mock_urlretrieve, temp_dir):
#         """Тест загрузки с использованием SetupSettings"""
#         settings = SetupSettings(
#             remote_files_urls=[
#                 "https://example.com/ru_rag_test_dataset/file1.txt",
#                 "https://example.com/RuBQ/test.json",
#             ]
#         )
#         mock_urlretrieve.return_value = (None, None)
#
#         with patch("src.fetch_datasets.print") as _:
#             extract_and_download(settings=settings)
#
#         assert mock_urlretrieve.call_count == 2
#         calls = mock_urlretrieve.call_args_list
#         # Проверяем, что файлы сохраняются в правильные подпапки
#         assert "ru_rag_test_dataset" in str(calls[0][0][1])
#         assert "RuBQ" in str(calls[1][0][1])
#
#     def test_empty_settings_error(self):
#         """Тест обработки пустых настроек"""
#         settings = SetupSettings(remote_files_urls=[])
#
#         with pytest.raises(SystemExit) as exc_info:
#             extract_and_download(settings=settings)
#
#         assert exc_info.value.code == 1
#
#
# class TestGithubGetFirst1000Files:
#     """Тесты для функции github_get_first_1000_files"""
#
#     @patch("src.fetch_datasets.requests.get")
#     def test_successful_github_api_response(self, mock_get):
#         """Тест успешного ответа от GitHub API"""
#         mock_response = mock_get.return_value
#         mock_response.status_code = 200
#         mock_response.json.return_value = [
#             {
#                 "name": "file1.py",
#                 "type": "file",
#                 "download_url": "https://raw.githubusercontent.com/user/repo/main/file1.py",
#             },
#             {
#                 "name": "file2.py",
#                 "type": "file",
#                 "download_url": "https://raw.githubusercontent.com/user/repo/main/file2.py",
#             },
#             {"name": "subdir", "type": "dir"},
#         ]
#
#         result = github_get_first_1000_files("https://api.github.com/repos/user/repo/contents/")
#
#         assert result is not None
#         assert len(result) == 2
#         assert result[0]["name"] == "file1.py"
#         assert result[1]["name"] == "file2.py"
#         mock_get.assert_called_once()
#
#     @patch("src.fetch_datasets.requests.get")
#     def test_github_api_error_response(self, mock_get):
#         """Тест обработки ошибки от GitHub API"""
#         mock_response = mock_get.return_value
#         mock_response.status_code = 404
#
#         result = github_get_first_1000_files("https://api.github.com/repos/user/repo/contents/")
#
#         assert result is None
#         mock_get.assert_called_once()
#
#     @patch("src.fetch_datasets.requests.get")
#     def test_github_api_max_files_warning(self, mock_get):
#         """Тест предупреждения о максимальном количестве файлов"""
#         mock_response = mock_get.return_value
#         mock_response.status_code = 200
#         # Создаем 1000 файлов для проверки предупреждения
#         mock_response.json.return_value = [
#             {
#                 "name": f"file{i}.py",
#                 "type": "file",
#                 "download_url": f"https://raw.githubusercontent.com/user/repo/main/file{i}.py",
#             }
#             for i in range(1000)
#         ]
#
#         with patch("src.fetch_datasets.logger") as mock_logger:
#             result = github_get_first_1000_files("https://api.github.com/repos/user/repo/contents/")
#
#             assert result is not None
#             assert len(result) == 1000
#             mock_logger.warning.assert_called_once_with(
#                 "Превышено максимальное количество файлов на директорию. Используй Git Trees API"
#             )
#
#     @patch("src.fetch_datasets.requests.get")
#     def test_github_api_network_error(self, mock_get):
#         """Тест обработки сетевой ошибки при запросе к GitHub API"""
#         mock_get.side_effect = Exception("Network error")
#
#         with patch("src.fetch_datasets.logger") as mock_logger:
#             result = github_get_first_1000_files("https://api.github.com/repos/user/repo/contents/")
#
#             assert result is None
#             mock_logger.error.assert_called_once()
#
#     @patch("src.fetch_datasets.requests.get")
#     def test_github_api_with_mixed_content(self, mock_get):
#         """Тест ответа API с файлами и директориями"""
#         mock_response = mock_get.return_value
#         mock_response.status_code = 200
#         mock_response.json.return_value = [
#             {
#                 "name": "README.md",
#                 "type": "file",
#                 "download_url": "https://raw.githubusercontent.com/user/repo/main/README.md",
#             },
#             {"name": "src", "type": "dir"},
#             {
#                 "name": "config.json",
#                 "type": "file",
#                 "download_url": "https://raw.githubusercontent.com/user/repo/main/config.json",
#             },
#         ]
#
#         result = github_get_first_1000_files("https://api.github.com/repos/user/repo/contents/")
#
#         assert result is not None
#         assert len(result) == 2  # Только файлы, без директорий
#         assert result[0]["name"] == "README.md"
#         assert result[1]["name"] == "config.json"
#
#
# class TestEnvironmentVariable:
#     """Отдельные тесты для проверки переменной окружения"""
#
#     def test_test_env_visible_in_subprocess(self, temp_dir):
#         """Проверяет, что test=true доступно в подпроцессах"""
#
#         result = subprocess.run(
#             [sys.executable, "-c", "import os; print(os.getenv('test'))"],
#             capture_output=True,
#             text=True,
#             env={**os.environ},
#             check=True,
#         )
#         assert result.stdout.strip() == "true"
#
#     def test_env_not_persisted_after_test(self, monkeypatch):
#         """Проверяет, что фикстура не загрязняет глобальное окружение"""
#         monkeypatch.setenv("test", "true")
#         assert os.getenv("test") == "true"
#         # После выхода из теста с monkeypatch значение должно быть откачено
#         # (это проверяется автоматически pytest)
