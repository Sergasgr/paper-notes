# Uni-LoRA: One Vector is All You Need (NeurIPS 2025, University of Connecticut / NEC Labs America — Li et al.)

**Pregunta con la que lo leí:** ¿Cómo unifica Uni-LoRA las variantes ultraligeras de LoRA bajo una misma matriz de proyección, y cómo consigue entrenar todo el adaptador con un único vector sin deformar la geometría del aprendizaje?

**Problema:** Para comprimir LoRA aún más, variantes como Tied-LoRA, VeRA, LoRA-XS o VB-LoRA llevan los parámetros entrenables a un subespacio de menor dimensión, pero la mayoría hace proyecciones **locales** (capa por capa) que impiden compartir información por toda la red, asigna dimensiones fijas a cada capa como si todas importaran igual, y usa proyecciones que **no conservan las distancias** (no isométricas), deformando el "paisaje de optimización".

**Idea clave:** Aplanar todas las matrices $B$ y $A$ de todas las capas en un vector gigante $\theta_D$ y reconstruirlo desde un único vector minúsculo entrenable $\theta_d$ mediante una proyección **congelada, global, uniforme e isométrica** ($\theta_D = P\theta_d$): es repartir al azar los $D$ parámetros de toda la red en $d$ grupos equilibrados que comparten el mismo valor.

**Método:**
Uni-LoRA muestra que muchas variantes de LoRA se pueden expresar en un mismo marco: una "matriz de proyección" $P$ que conecta su pequeño subespacio con el espacio original de LoRA. Se extraen todas las matrices $B$ y $A$ de las distintas capas y se concatenan y aplanan en un único vector columna $\theta_D$ ($D = L(m+n)r$), y solo se entrena $\theta_d$ ($d \ll D$):

```math
\theta_D = P\,\theta_d, \qquad P \in \mathbb{R}^{D \times d}, \qquad P^\top P = I_d
```

* **El marco unificado:** LoRA estándar es $P = I_D$ (isométrica pero sin compresión). VeRA y Tied-LoRA usan una $P$ diagonal por bloques (local por capa) y no uniforme (reparto distinto entre $m$ y $r$); LoRA-XS y FourierFT también son locales; VB-LoRA es global, pero su $P$ se aprende y no es isométrica. A partir de ahí, los autores diseñan una $P$ con tres propiedades: **globalidad, uniformidad (equilibrio de carga) e isometría**.
* **Construcción de $P$:** $P$ no se entrena: se genera aleatoriamente y se congela. Cada fila es un vector *one-hot* que asigna cada parámetro de $\theta_D$ a una posición de $\theta_d$, seguido de una normalización por columnas ($1/\sqrt{n_j}$, con $n_j$ el número de parámetros del grupo $j$) para que los grupos que por azar recibieron más parámetros no dominen el entrenamiento. Como cada fila tiene un solo valor no nulo, las columnas son ortogonales y de norma 1, así que $P^\top P = I_d$ y las distancias se conservan. La regla estricta es que todos los parámetros de un mismo grupo comparten exactamente el mismo valor durante todo el entrenamiento: actúan como un bloque y copian la actualización de su posición en $\theta_d$.
* **Implementación:** $P$ nunca se materializa: cada capa guarda los índices (generados desde una semilla) y los factores de normalización, reconstruye $A$ y $B$ al vuelo indexando $\theta_d$ y calcula $(xA)B$ sin construir $\Delta W$. Proyectar cuesta $\mathcal{O}(D)$ (frente a $\mathcal{O}(D \log d)$ de Fastfood), y al final basta con guardar **solo $d + 1$ números**: $\theta_d$ y la semilla.

**Resultado clave:**
* **GLUE (RoBERTa-large, $r=4$):** **88.3** de media con **0.023M** parámetros, frente a LoRA (87.8 con 0.786M, usando menos del 3% de sus parámetros), VB-LoRA (88.2) y FourierFT (88.0 con 0.048M).
* **Matemáticas (Gemma-7B, MetaMathQA, $d$ = 0.52M):** **75.59%** en GSM8K y **28.94%** en MATH con el 0.26% de los parámetros de LoRA, por encima de *full fine-tuning* (71.34% / 22.74%).
* **Visión (ViT-Base, 8 datasets):** **85.15%** con 72K parámetros, frente a 77.75% de FourierFT con los mismos parámetros y 80.29% con 239K.
* **Mis experimentos (`FLUX.2-klein`, $r=4$):** barrido de `theta_d_length`: 100k → 0.3912 / drift 0.0383; 250k → 0.4394 / 0.0572; **500k → 0.6562 / 0.2055**; 750k → 0.6597 / 0.2359. Rendimientos decrecientes: subida fuerte hasta 500k y meseta después. Con 500k: 9,598 MB, 1,042 s y un checkpoint de **1.9 MB**, frente a LoRA $r=32$ (0.6815 / 0.2721, 146.3 MB). En el benchmark oficial (L40S): **0.6573** / drift 0.2300.

**Limitaciones:**
* **Qué no prueban:** NLP y visión de escala pequeña a mediana (hasta 13B), ni modelos gigantes ni difusión.
* **Supuestos:** necesita *learning rates* bastante mayores que LoRA (p. ej. `8e-4` frente a `4e-4` en LLaMA2, hasta `5e-3`–`2e-2` en GLUE y ViT), y la calidad cae bruscamente si $d$ es demasiado pequeño. En FLUX, subir $r$ con $d$ fijo **empeora**: $r=32$ da 0.4462 frente a 0.6562 con $r=4$ y gasta 410 MB más. Con $r=4$, $\theta_D$ tiene unos 4.8M parámetros y cada valor de $\theta_d$ se comparte entre unos 10; con $r=32$ son unos 38.3M y cada valor se comparte entre unos 77. El paper no estudia $r$ con $d$ fijo, así que esta conclusión es mía.
* **Coste:** aunque el optimizador y el checkpoint son minúsculos, en cada *forward/backward* hay que reconstruir $A$ y $B$ completas de rango $r$, así que la VRAM y el tiempo por paso son prácticamente los de un LoRA del mismo rango (7,212 MiB frente a 7,044 MiB en LLaMA2-7B).

**Conexión:** `#peft` `#lora` `#weight-sharing` `#isometric-projection` `#vera` `#vb-lora` `#flux2-klein` (unifica VeRA, Tied-LoRA, LoRA-XS, VB-LoRA y FourierFT; comparte parámetros al azar entre toda la red, a diferencia de TinyLoRA, donde subir $r$ no añade compartición porque $r$ es el tamaño del subespacio SVD congelado).

**Preguntas**
1. **Conceptual:** *¿Qué critica Uni-LoRA de las proyecciones de VeRA, Tied-LoRA o VB-LoRA, y qué tres propiedades exige a su $P$?*
   * **Respuesta:** Que son locales (capa por capa, sin compartir parámetros entre capas y con el mismo tamaño para todas aunque unas importen más), no uniformes ($m$ frente a $r$) y no isométricas, lo que deforma las distancias del espacio de optimización (VB-LoRA es global, pero aprende su $P$ y tampoco es isométrica). Uni-LoRA exige **globalidad** (compartir entre todas las capas y matrices), **uniformidad** (grupos de tamaño parecido) e **isometría** ($P^\top P = I_d$).
2. **Matemática / Diseño:** *¿Por qué la $P$ de Uni-LoRA es una isometría?*
   * **Respuesta:** Cada fila es *one-hot*, así que dos columnas distintas nunca tienen un valor no nulo en la misma fila: $`[P^\top P]_{j,k} = 0`$ si $j \ne k$. Cada entrada no nula de la columna $j$ vale $1/\sqrt{n_j}$, así que $`[P^\top P]_{j,j} = n_j \cdot (1/\sqrt{n_j})^2 = 1`$. Por tanto $P^\top P = I_d$ y $`\|P(x - y)\| = \|x - y\|`$.
3. **Crítica / Trade-offs:** *¿Por qué en `FLUX.2-klein` subir $r$ de 4 a 32 empeoró el DINOv2 (0.6562 → 0.4462), y por qué Uni-LoRA no ahorra VRAM frente a un LoRA del mismo rango?*
   * **Respuesta:** Porque con $d$ fijo, subir $r$ no añade parámetros entrenables: solo hace que muchos más parámetros de LoRA caigan en el mismo grupo (~77 por valor de $\theta_d$ frente a ~10), así que la actualización queda más atada. Y no ahorra VRAM porque las $A$ y $B$ completas se reconstruyen en cada paso; el ahorro está en los estados del optimizador y en el checkpoint ($d+1$ números).

**Flashcards**
1. **P:** ¿Cuál es la ecuación central de Uni-LoRA?
   **R:** $\theta_D = P\theta_d$: $\theta_D$ son todas las $B$ y $A$ de LoRA aplanadas y concatenadas, $\theta_d$ ($d \ll D$) es el único vector entrenable y $P$ es la proyección congelada.
2. **P:** ¿Cómo se construye $P$?
   **R:** Cada fila es un *one-hot* aleatorio que asigna un parámetro de $\theta_D$ a una posición de $\theta_d$; después cada columna $j$ se normaliza por $1/\sqrt{n_j}$. Resultado: $d$ grupos de tamaño parecido que comparten valor, con $P^\top P = I_d$.
3. **P:** ¿Qué hay que guardar al terminar el entrenamiento?
   **R:** Solo $d + 1$ números: el vector $\theta_d$ y la semilla con la que se regeneran los índices de $P$ (1.9 MB con 500k en `FLUX.2-klein`).
4. **P:** ¿Qué `theta_d_length` elegí en `FLUX.2-klein` y por qué?
   **R:** 500k: casi el mismo DINOv2 que 750k (0.6562 frente a 0.6597) con menos parámetros. La diferencia de drift entre ambos (0.03) es del tamaño de la variación entre runs, así que no se justifica solo por el drift.
