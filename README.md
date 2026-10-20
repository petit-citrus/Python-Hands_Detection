# Python-Hands_Detection

L'objectif est de développer une application Python utilisant une webcam pour détecter les mains en temps réel et afficher leur squelette.

## Technologies utilisées

- **Python** (3.11 ou 3.12) : langage de programmation.
- **OpenCV** : capture et affichage des images de la webcam.
- **MediaPipe** (`0.10.14`) : détection des mains et extraction des 21 points clés (landmarks).
- **NumPy** : manipulation des données numériques.

## Organisation du projet

```text
├── main.py            # Capture vidéo, interface et boucle principale
├── hand_analyzer.py   # Module de détection et traçage du squelette des mains
├── pyproject.toml     # Configuration du projet et dépendances (uv)
└── requirements.txt   # Liste des dépendances (pip)