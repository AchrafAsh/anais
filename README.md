# ANAIS

### Installation

```{bash}
$ git clone https://github.com/AchrafAsh/anais.git
$ cd anais
$ pip install -r requirements.txt
```

Launch app on AWS:
```{bash}
$ gunicorn --chdir app --workers=2 --bind=0.0.0.0:5000 main:app
```

### Utilisation

Les modèles sont dans le dossier `models`

### Documentation
```
$ pdoc **/*.py
```
