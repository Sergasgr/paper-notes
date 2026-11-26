# DoRA: Weight-Decomposed Low-Rank Adaptation (ICML 2024 Oral, NVIDIA / HKUST — Liu et al.)

**Pregunta con la que lo leí:** ¿Por qué separar magnitud y dirección hace que DoRA aprenda de forma más parecida al *fine-tuning* completo que LoRA, y cuánto cuesta eso en entrenamiento?

**Problema:** LoRA modifica los pesos sumando una matriz de bajo rango ($W_0 + BA$): al usar una suma simple, encadena la **dirección** (el patrón, la proporción entre los números) y la **magnitud** (la fuerza de ese patrón), de modo que si necesita cambiar mucho la dirección, la suma altera también la magnitud. Según el paper, esa rigidez, y no solo el bajo número de parámetros, es un límite de LoRA frente al *fine-tuning* completo (FT).

**Idea clave:** Escribir cada peso como magnitud × dirección y entrenar cada parte por separado: la dirección con LoRA (es la parte con más parámetros) y la magnitud con un vector $m$ de un valor por columna, para que se ajusten de forma independiente, como en FT.

**Método:**
Magnitud y dirección: toma una columna de la matriz de pesos (en PyTorch, una fila de `weight`: los pesos que llegan a una neurona de salida), por ejemplo $[2, 4]$. La dirección es la proporción entre los números (el segundo es el doble del primero): es el patrón que detecta esa neurona. La magnitud es su norma ($\sqrt{2^2 + 4^2} \approx 4.5$): la fuerza con la que reacciona a ese patrón. $[20, 40]$ tiene la misma dirección y 10 veces más magnitud.

$$
W' = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c}
$$

* **Lo que encuentran:** en FT, magnitud y dirección cambian de forma bastante independiente (correlación negativa, −0.62): puede hacer un gran cambio de dirección casi sin tocar la magnitud, o al revés. En LoRA están atadas (correlación positiva, 0.83): cuando cambia mucho la dirección, la magnitud cambia con ella. DoRA queda en −0.31, con un patrón más parecido al de FT. Para pasar la proporción de $[2, 4]$ a algo como $[8, 1]$, LoRA tiene que sumar valores grandes, que también cambian el tamaño, y con tan pocos parámetros no puede compensarlo.
* **La solución en tres pasos:** (1) dirección: $W_0 + BA$, con $A$ y $B$ como en LoRA; (2) normalización: se divide cada columna por su norma ($\|\cdot\|_c$), de forma que cada columna mide 1 y queda solo la dirección; (3) magnitud: se multiplica por $m$, un vector entrenable con un valor por columna que decide la fuerza de cada neurona. Al principio $m = \|W_0\|_c$ y $B = 0$, así que $W' = W_0$. Se entrenan $A$, $B$ y $m$; como $m$ es solo un vector, apenas añade parámetros (un 0.01% más que LoRA en el paper). Además, la normalización cambia el gradiente de la dirección: lo reescala por $m / \|W_0 + BA\|_c$ y lo proyecta fuera de la dirección actual del peso, lo que acerca su covarianza a la identidad y estabiliza la optimización (como en *weight normalization*).
* **Truco de memoria y *merge*:** la norma $\|W_0 + BA\|_c$ depende de $A$ y $B$, y propagar el gradiente a través de ella gasta mucha memoria. DoRA la trata como una constante en el *backward* (`detach`): se recalcula en cada paso, pero no recibe gradiente. El paper reporta un 24.4% menos de memoria en LLaMA-7B (de 37.3 GB a 28.2 GB, con 78.1% frente a 78.3% de precisión) y un 12.4% en VL-BART (sin perder precisión). Al terminar, magnitud y dirección se fusionan en una sola matriz, así que en inferencia no añade latencia. En `peft` no es un método aparte: se activa con `use_dora=True` en `LoraConfig`, y $m$ se guarda como `lora_magnitude_vector`.

**Resultado clave:**
* **Sentido común (8 benchmarks, mismo rango que LoRA):** mejora a LoRA en todos los modelos: **+3.7** en LLaMA-7B (78.4% frente a 74.7%), +1.0 en LLaMA-13B, +2.1 en LLaMA2-7B y **+4.4** en LLaMA3-8B (85.2% frente a 80.8%). Con la mitad del rango (DoRA†) también supera a LoRA: +2.8, +1.0, +2.9 y +4.2; en LLaMA2-7B, DoRA† incluso supera al DoRA completo (80.5% frente a 79.7%).
* **Multimodal y diálogo:** +0.9 / +1.9 en comprensión imagen-texto / vídeo-texto (VL-BART), +0.7 en *visual instruction tuning* (LLaVA-1.5-7B: 67.6% frente a 66.9% de LoRA y 66.5% de FT) y +0.4 / +0.3 en MT-Bench (LLaMA / LLaMA2-7B). Combinado con VeRA para la dirección (DVoRA) también mejora a VeRA en MT-Bench (6.0 frente a 5.5 en LLaMA2-7B).
* **En el benchmark image-gen de `peft` (L40S):** DoRA $r=64$ entrena en 1,436 s frente a 717 s de LoRA $r=64$ y usa 12,566 MiB frente a 10,856 MiB.

**Limitaciones:**
* **Qué no prueban:** casi todo son LLMs y modelos de visión-lenguaje (LLaMA, VL-BART, LLaVA). En difusión solo hay un experimento cualitativo (DreamBooth en SDXL con iconos 3D y Lego, sin métricas), donde DoRA capta mejor rasgos propios del sujeto, como el logo de Lego. El repositorio oficial marca la difusión como experimental: LoRA parece converger más rápido que DoRA y puede necesitar hiperparámetros distintos.
* **Supuestos:** la explicación (que imitar el patrón de FT, con magnitud y dirección independientes, es lo que mejora el aprendizaje) se apoya en el análisis de correlación de la matriz *query* en varias capas y en el análisis del gradiente; es una hipótesis bien argumentada más que una prueba causal. De hecho, el propio paper reconoce que cuando FT rinde peor que LoRA (LLaVA), la mejora de DoRA es menor. El `detach` de la norma es una aproximación del gradiente real, aceptada porque la diferencia es despreciable.
* **Coste:** durante el entrenamiento es más caro que LoRA, porque en cada paso construye la matriz completa $W_0 + BA$ para calcular la norma: en el benchmark de `peft`, unas **2x** de tiempo y +1.7 GB de memoria con $r=64$. Solo tras el *merge* el coste en inferencia es cero (sin *merge*, hay cálculo extra).

**Conexión:** `#peft` `#lora` `#magnitude-direction` `#weight-decomposition` `#flux2-klein` (baseline fuerte en HiRA, HOFT y MonteCLoRA; SHOFT toma de DoRA el vector de magnitud entre sus dos rotaciones; como HiRA y LoHa, pierde parte del atajo de LoRA al construir $W_0 + BA$ completa; los autores agradecen a Bossan y al equipo de Hugging Face su ayuda para integrarlo en `peft`).

**Preguntas**
1. **Conceptual:** *¿Qué descubre el análisis de DoRA al comparar LoRA con el fine-tuning completo?*
   * **Respuesta:** Que en FT la magnitud y la dirección de los pesos cambian de forma casi independiente (correlación negativa, −0.62), mientras que en LoRA cambian juntas (correlación positiva, 0.83). LoRA solo suma $BA$, así que no puede hacer un gran cambio de dirección sin alterar la magnitud. Según el paper, aprender las dos cosas a la vez con matrices de bajo rango es difícil, y ese es un límite de LoRA además del número de parámetros.
2. **Matemática / Diseño:** *¿Qué hace cada término de $W' = m \cdot (W_0 + BA) / \|W_0 + BA\|_c$ y cómo se inicializa para no cambiar el modelo al principio?*
   * **Respuesta:** $W_0 + BA$ cambia la dirección con LoRA; dividir por la norma de cada columna deja solo la dirección (cada columna mide 1); $m$ reescala cada columna con la fuerza que necesite la tarea. Con $m = \|W_0\|_c$ y $B = 0$ al inicio, $W' = W_0 / \|W_0\|_c \cdot \|W_0\|_c = W_0$.
3. **Crítica / Trade-offs:** *Si $m$ apenas añade parámetros, ¿por qué DoRA tarda el doble que LoRA en el benchmark de `peft`, y cómo limita ese coste el `detach`?*
   * **Respuesta:** Porque para normalizar necesita la norma por columnas de $W_0 + BA$, lo que obliga a construir la matriz completa en cada paso, algo que LoRA nunca hace. El `detach` evita además guardar el grafo de la norma para el *backward* (−24.4% de memoria en LLaMA), a cambio de un gradiente aproximado con un impacto despreciable en precisión. En inferencia, tras el *merge*, el coste es cero.

**Flashcards**
1. **P:** ¿Qué son la dirección y la magnitud de una columna de pesos como $[2, 4]$?
   **R:** La dirección es la proporción entre los números (el patrón que detecta la neurona); la magnitud es su norma ($\approx 4.5$), la fuerza con la que reacciona. $[20, 40]$ tiene la misma dirección y 10 veces más magnitud.
2. **P:** ¿Cuál es la ecuación de DoRA?
   **R:** $W' = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c}$: la dirección se aprende con LoRA, se normaliza por columnas y se multiplica por el vector entrenable de magnitud $m$.
3. **P:** ¿Cuál es el truco de memoria de DoRA?
   **R:** Tratar $\|W_0 + BA\|_c$ como una constante en el *backward* (`detach`): −24.4% de memoria en LLaMA con una diferencia de precisión despreciable.
4. **P:** ¿Cómo se usa DoRA en `peft`?
   **R:** Con `use_dora=True` en `LoraConfig`; el vector $m$ se guarda como `lora_magnitude_vector`.
