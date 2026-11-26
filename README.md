# paper-notes

mis resúmenes de papers de IA/ML.

## Estructura de cada nota

```
# <Título> (<año>, <lab o autores>)
**Pregunta con la que lo leí:**
**Problema:** qué no funcionaba antes, en 1–2 frases
**Idea clave:** una frase que entienda alguien de mi clase
**Método:** la ecuación o el diagrama central, escrito por mí
**Resultado clave:** el número que importa y contra qué baseline (+ mis experimentos, si los hay)
**Limitaciones:** qué no prueban, supuestos, coste
**Conexión:** tags + relación con TFG / CodeAlign / Distill / peft / pregunta del examen
**Preguntas:** 3 de entrevista (conceptual, matemática/diseño, crítica/trade-offs), con respuesta
**Flashcards:** 3–5, en forma de pregunta (P/R)
```

## Carpetas

```
eng/        notas en inglés
esp/        notas en español
  peft/     métodos PEFT (+ mis experimentos en el benchmark image-gen de peft)
  rl/
  ...
```

## Experimentos en `FLUX.2-klein`

Las notas de `peft/` incluyen mis runs en el benchmark de comparación de métodos image-gen de [huggingface/peft](https://github.com/huggingface/peft) (`FLUX.2-klein`, sujeto *few-shot* con 2 imágenes de validación y 4 de test, 750 pasos):

- **DINOv2 similarity (test):** similitud coseno DINOv2 con las imágenes de test (fidelidad al sujeto; más alto = mejor).
- **Drift:** cuánto cambian las imágenes para prompts sin relación (p. ej. *"an astronaut riding a horse"*) respecto al modelo base con la misma semilla: (1 − similitud coseno media de DINOv2) / 2; 0 = imágenes idénticas.
- **Baseline:** LoRA r=32 (DINOv2 0.6815 / drift 0.2721 en mi RTX 5070 Ti; 0.6972 / 0.2620 en la L40S del benchmark oficial).
