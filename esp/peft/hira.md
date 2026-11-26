# HiRA: Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models (ICLR 2025, SUSTech / Surrey / ByteDance — Huang et al.)

**Pregunta con la que lo leí:** ¿Cómo rompe HiRA el límite de rango de LoRA con exactamente los mismos parámetros, y qué coste y sensibilidad al *learning rate* tiene al llevarlo a `FLUX.2-klein`?

**Problema:** LoRA crea dos matrices pequeñas ($L_1$ y $L_2$, como $A$ y $B$) y suma su producto a los pesos congelados: al sumar directamente intenta crear información "desde cero" (como pintar con brocha gorda) y la actualización queda limitada a rango $r$, lo que se queda corto en tareas complejas como razonamiento o matemáticas; subir $r$ para ganar expresividad sube los parámetros en la misma proporción.

**Idea clave:** En vez de sumar el producto $L_1 L_2$, HiRA lo multiplica elemento a elemento (producto de Hadamard $\odot$) por la matriz de pesos original, de modo que $L_1 L_2$ actúa como un **ecualizador o mapa de multiplicadores** sobre lo que el modelo ya sabe y la actualización pasa a ser de **alto rango** con los mismos parámetros que LoRA.

**Método:**
HiRA es una evolución de LoRA. Al igual que LoRA crea dos matrices ($L_1$ y $L_2$), pero en vez de sumar simplemente su producto a la matriz de pesos congelada, suma el producto Hadamard entre la matriz de pesos original y $L_1 L_2$:

$$
W' = W_0 + W_0 \odot (L_1 L_2), \qquad \text{rango}\big(W_0 \odot (L_1 L_2)\big) \le \text{rango}(W_0) \cdot r
$$

* **Por qué rompe el límite de rango:** el límite $r$ es del producto $L_1 L_2$, no de cada matriz por separado. En $L_1 L_2$ muchas filas son múltiplos o combinaciones lineales de otras (información redundante). Las filas de $W_0$ son distintas entre sí: al multiplicar elemento a elemento, cada fila se multiplica por números diferentes y deja de ser múltiplo de las demás, rompiendo la dependencia lineal. En el paper (LLaMA-3-8B, $r=32$) el rango de $\Delta W$ pasa de **32** con LoRA a **2837** con HiRA. Eso es lo que el paper llama "alto rango".
* **Subirse a los hombros del modelo:** si un peso de $W_0$ es grande (representa un concepto que el modelo ya domina), al hacer el Hadamard su actualización cobra mucha más importancia: $L_1 L_2$ no crea ideas nuevas ni tiene que generar un número grande desde cero, solo decide qué amplificar y qué apagar. Los gradientes de $L_1$ y $L_2$ también quedan modulados por $W_0$. Usar $W_0$ importa: con una matriz aleatoria $R$ en su lugar, la media en sentido común cae de 86.72% a 37.63%.
* **Inicialización y *merge*:** $L_1$ empieza a 0 (y $L_2$ con Kaiming), así que $\Delta W = 0$ al principio. Al terminar se fusiona con los pesos ($W' = W_0 \odot (1 + L_1 L_2)$), así que en inferencia no añade coste. El paper también prueba HiLoRA, que suma ambas ramas: $\Delta W = W_0 \odot (L_1 L_2) + L_1' L_2'$.

**Resultado clave:**
* **Sentido común (8 benchmarks, $r=32$):** en LLaMA-3-8B logra **86.72%** de media frente a DoRA (**85.20%**), LoRA (**80.79%**) y MoRA (78.63%); con la mitad de parámetros ($r=16$) aún saca 86.08%. En LLaMA-2-7B: 81.42% frente a 79.69% (DoRA) y 77.61% (LoRA).
* **Matemáticas (MetaMath → GSM8K, $r=32$):** en LLaMA-3-8B, **70.81%** frente a 65.89% (LoRA) y 66.12% (DoRA); en LLaMA-2-7B, **46.85%** frente al **15.16%** de LoRA.
* **Mis experimentos (`FLUX.2-klein`, $r=32$, 38,338,560 parámetros = 0.98%, igual que LoRA):** con el lr por defecto (`1e-4`, 750 pasos) no llega a converger: **DINOv2 = 0.5550**, drift 0.1454, frente a LoRA (**0.6815** / 0.2721). Un sweep con Optuna en el rango del paper (`[1e-4, 2e-4]`) dio `0.000153`; probé `5e-4` (fuera de la tabla de hiperparámetros, aunque el texto principal menciona `0.001`) y dio los mejores resultados: en el benchmark oficial (L40S), **0.7063** / drift 0.2607 frente a **0.6972** / 0.2620 de LoRA.

**Limitaciones:**
* **Qué no prueban:** solo LLMs en tareas de lenguaje (sentido común, matemáticas y diálogo), nada de difusión ni generación de imágenes.
* **Supuestos (*learning rate*):** necesita un lr bastante mayor que el estándar de LoRA para converger en pocos pasos (`5e-4` en FLUX, `0.001` en el paper). Con `1e-4` el drift bajo solo indicaba que el modelo había aprendido poco; con `5e-4` sube al nivel de LoRA (0.2626). Bossan avisó del peligro de tunear con el test (sobreajustar al test set, que deja de ser una estimación fiable), pero el dataset es muy pequeño (2 imágenes de validación y 4 de test) y la validación es muy ruidosa.
* **Coste:** no puede usar el atajo de bajo rango de LoRA (multiplicar la entrada por $L_1$ y luego por $L_2$ sin construir nunca la matriz): en cada paso construye la matriz completa $W_0 \odot (L_1 L_2)$ y hace una multiplicación de tamaño completo. En FLUX tarda unas **2.8x** más que LoRA (3,229 s frente a 1,161 s en la misma GPU; 2,140 s frente a 759 s en la L40S) y gasta algo más de memoria (10,914 MB frente a 10,366 MB). El paper, en cambio, reporta menos memoria y tiempo que LoRA en LLaMA-3-8B (61.49 GB / ~14 h frente a 65.48 GB / ~15 h), así que el sobrecoste depende de la implementación y de cuánto pesen esas capas en el entrenamiento.

**Conexión:** `#peft` `#hadamard` `#high-rank` `#lora` `#flux2-klein` `#learning-rate` (usa la misma desigualdad de rango que LoHa, que multiplica dos productos de bajo rango entre sí en vez de por $W_0$; como LoHa y FourierFT, pierde el atajo de LoRA y necesitó subir el lr para superar a LoRA en el benchmark).

**Preguntas**
1. **Conceptual:** *¿Cómo consigue HiRA una actualización de rango alto (2837) con los mismos parámetros que un LoRA de $r=32$?*
   * **Respuesta:** Multiplicando elemento a elemento $L_1 L_2$ por los pesos congelados: $\Delta W = W_0 \odot (L_1 L_2)$. Como $\text{rango}(P \odot Q) \le \text{rango}(P) \cdot \text{rango}(Q)$, la cota pasa de $r$ a $\text{rango}(W_0) \cdot r$. Intuitivamente, las filas redundantes de $L_1 L_2$ se multiplican por filas distintas de $W_0$ y dejan de ser combinaciones lineales unas de otras. Los parámetros entrenables siguen siendo los de $L_1$ y $L_2$.
2. **Matemática / Diseño:** *¿Por qué usar $W_0$ en el Hadamard y no cualquier otra matriz fija de rango completo?*
   * **Respuesta:** Porque $W_0$ aporta el conocimiento preentrenado: amplifica los pesos que ya son importantes y modula los gradientes de $L_1$ y $L_2$ hacia direcciones relevantes. Con una matriz aleatoria $R$ la media cae de 86.72% a 37.63%. Además, $W_0$ ya está en memoria, así que no hay que guardar otra matriz, y el *merge* es directo: $W' = W_0 \odot (1 + L_1 L_2)$.
3. **Crítica / Trade-offs:** *Si HiRA tiene los mismos parámetros que LoRA y coste cero en inferencia, ¿por qué tarda ~2.8x más en `FLUX.2-klein` y por qué necesitó subir el lr?*
   * **Respuesta:** Porque pierde el atajo asociativo de LoRA ($x L_1 L_2$ como dos multiplicaciones pequeñas): el Hadamard con $W_0$ obliga a materializar en cada paso la matriz densa $W_0 \odot (L_1 L_2)$ y a hacer una multiplicación de tamaño completo. Con `1e-4` no convergía en 750 pasos (DINOv2 0.5550 y un drift bajo por aprender poco); con `5e-4` superó a LoRA (0.7063 frente a 0.6972 en la L40S). El *merge* solo elimina el coste en inferencia, no en entrenamiento.

**Flashcards**
1. **P:** ¿Cuál es la ecuación de HiRA y qué cota de rango alcanza frente a LoRA?
   **R:** $W' = W_0 + W_0 \odot (L_1 L_2)$. LoRA está acotado por $\text{rango}(\Delta W) \le r$; HiRA por $\text{rango}(W_0) \cdot r$ (en el paper, 2837 frente a 32).
2. **P:** ¿Cuál es la diferencia intuitiva entre LoRA y HiRA?
   **R:** LoRA dibuja un cuadro nuevo en un papel transparente con muy poca pintura y lo pega encima (suma un parche desde cero). HiRA usa esa poca pintura como un filtro inteligente que altera los colores del cuadro ya pintado (multiplica sobre lo que el modelo ya sabe).
3. **P:** ¿Por qué HiRA es más lento que LoRA en entrenamiento pese a tener los mismos parámetros?
   **R:** Porque no puede usar el atajo de bajo rango: el Hadamard obliga a construir en cada paso la matriz completa $W_0 \odot (L_1 L_2)$ y hacer una multiplicación de tamaño completo (~2.8x en `FLUX.2-klein`).
4. **P:** ¿Qué pasó en `FLUX.2-klein` con lr `1e-4` frente a `5e-4`?
   **R:** Con `1e-4` no convergía en 750 pasos (DINOv2 0.5550, drift 0.1454 por aprender poco); con `5e-4` superó a LoRA (0.7063 / 0.2607 frente a 0.6972 / 0.2620 en la L40S).
