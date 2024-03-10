Основное исследование приведено в файле main.ipynb

В файле main.py приведено приложение, которое упаковано в докер контейнер

Запуск докер контейнера:
```bash
docker build -t image_name .
docker run --rm -ti image_name
```

В контейнер на уровень с main.py загружены csv файлы из папки data.