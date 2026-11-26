# AdaMSS: Adaptive Multi-Subspace Approach for Parameter-Efficient Fine-Tuning (NeurIPS 2025, Peking University / UBC — Zheng et al.)

**Pregunta con la que lo leí:** ¿Cómo encuentra AdaMSS varios subespacios dentro de $W_0$ y entrena solo los más útiles, y cómo se compara con LoRA en `FLUX.2-klein` usando 13 veces menos parámetros?

**Problema:** LoRA, PiSSA, LoRA-GA y similares son métodos "de un solo subespacio": asumen que los cambios de pesos, los pesos o los gradientes viven en un único subespacio de baja dimensión, lo que obliga a elegir entre poca expresividad (un $r$ pequeño limita la adaptación) o muchos parámetros y memoria (un $r$ grande).

**Idea clave:** Agrupar las columnas de $W_0$ en varios subespacios pequeños e independientes (columnas que son combinaciones lineales unas de otras), entrenar una actualización ultraligera en cada uno ($W = W_0 + ABC$, con $A$ congelada) y, durante el entrenamiento, ir congelando los subespacios que menos aportan a la tarea.

**Método:**
AdaMSS agrupa en un mismo cluster las columnas de $W_0$ que son aproximadamente combinaciones lineales las unas de las otras (es decir, que están en el mismo subespacio de baja dimensión), separándolas de los grupos con los que tienen poca relación. Cada cluster es un subespacio independiente y recibe su propia actualización:

```math
W^{(k)} = W_0^{(k)} + A^{(k)} B^{(k)} C^{(k)}, \qquad k = 1, \dots, K
```

* **Encontrar los subespacios:** con la SVD truncada de $W_0$ (rango $R$, `r=100` en `peft`) se obtiene la representación de menor rango $Z^\star = VV^\top$ ($W_0 \approx W_0 Z^\star$), que sale aproximadamente diagonal por bloques. Como las columnas de un mismo cluster están repartidas por $W_0$, se reordenan para que queden juntas, así $Z$ queda diagonal por bloques, con un bloque por subespacio. El posible ruido (lo que queda fuera de los bloques y lo que no captura la SVD truncada) se separa en una matriz de residuos que se queda congelada.
* **Actualización por bloque ($A$, $B$, $C$):** en lugar de hacer actualizaciones pesadas dentro de cada cluster, se hace SVD sobre cada bloque. De ahí sale $A$, que se queda congelada (los "rieles" que marcan las direcciones del subespacio, así lo aprendido se inyecta en la dirección correcta), y el aprendizaje se comprime en dos matrices pequeñas: $B$, inicializada ortogonal con una descomposición QR para respetar la estructura de $A$ desde el principio, y $C$, que almacena "cuánto" corregir y se inicializa a 0 para que al principio $W = W_0$. Su tamaño lo controla `subspace_rank`.
* **Poda adaptativa (ASA, `use_asa=True`):** durante el entrenamiento se puntúa la importancia de $B$ y $C$ de cada bloque (magnitud y gradiente, suavizados con medias móviles). Cada `mask_interval` pasos (100 en `peft`) se hace un ranking: los mejores siguen entrenando y los peores se congelan. El número de bloques activos baja con una curva cúbica desde $K$ (en `init_warmup`) hasta $K_{\text{target}}$ (`asa_target_subspaces`, 50 por defecto, contado entre todas las capas) en `final_warmup`. Con `use_asa=False` no hay poda: es LoRA con $K$ bloques estructurados.

**Resultado clave:**
* **Imagen (ViT-Large, 7 datasets):** **+4.7%** de accuracy media frente a LoRA usando el **15.4%** de sus parámetros entrenables.
* **NLU (RoBERTa-Large, GLUE):** ~7% más que PiSSA con ~94.4% menos parámetros, y a la par con LoRA usando el 5.62% de sus parámetros.
* **Matemáticas (LLaMA 2-7B, todas las proyecciones):** **50.80%** en GSM8K frente a **42.30%** de LoRA (4M frente a 320M parámetros, un 1.25%), por encima incluso de *full fine-tuning* (49.05%); en MATH, 7.22% frente a 5.50%.
* **Mis experimentos (`FLUX.2-klein`, $K=10$, `r=100`, `subspace_rank=3`, `use_asa=False`):** con **2,905,054** parámetros (**0.075%**, 13 veces menos que LoRA) da **DINOv2 = 0.6790** / drift 0.2519, prácticamente igual que LoRA $r=32$ (0.6815 / 0.2721), con menos memoria (9,716 MB frente a 10,366 MB) y casi el mismo tiempo (1,181 s frente a 1,161 s). En el benchmark oficial (L40S) supera ligeramente a LoRA en las dos métricas: **0.7029** / 0.2513 frente a 0.6972 / 0.2620 (912 s frente a 759 s).

**Limitaciones:**
* **Qué no prueban:** clasificación de imágenes, NLU y matemáticas (modelos de hasta 7-8B), no difusión. La parte adaptativa no siempre compensa: AdaMSS<sub>base</sub> (sin poda) suele quedar igual o un poco por encima, a cambio de más parámetros entrenables.
* **Supuestos:** asume que las columnas de $W_0$ viven aproximadamente en varios subespacios independientes (lo apoya la estructura casi diagonal por bloques de $Z^\star$ en ViT, RoBERTa y LLaMA); la segmentación es un clustering aproximado. Con ASA, el calendario tiene que caber en el presupuesto de pasos: el default `final_warmup=1000` está por encima de los 750 pasos del benchmark, así que probé también `600` (0.6823 / 0.2416, frente a 0.6784 / 0.2548 con `1000`). La diferencia con `use_asa=False` es mínima, y Bossan prefirió mantener el experimento tal cual.
* **Coste:** la inicialización (SVD + clustering por capa) es más cara que la de PiSSA o LoRA-PRO (8.8 s frente a 2.3 s en LLaMA 2-7B). El checkpoint pesa 128.5 MB pese a tener 13 veces menos parámetros, porque guarda en fp32 las proyecciones congeladas de la SVD ($100 \times d_{\text{in}}$ por capa con `r=100`, unos 117 MB); los parámetros entrenables solo ocupan unos 11 MB.

**Conexión:** `#peft` `#subspace-segmentation` `#svd` `#adaptive-budget` `#adalora` `#flux2-klein` (primo de AdaLoRA: reutiliza el mismo `update_and_allocate`, pero poda bloques enteros de subespacio en vez de rangos dentro de una matriz. En el PR de resultados (#3641) añadí a `run.py` la llamada a `update_and_allocate(step)`, porque el callback de `peft` es para el `Trainer` de transformers y `run.py` usa su propio bucle; el checkpoint inflado por factores SVD congelados también pasa en TinyLoRA).

**Preguntas**
1. **Conceptual:** *¿Qué significa que LoRA o PiSSA sean métodos "de un solo subespacio" y qué gana AdaMSS usando varios?*
   * **Respuesta:** Que asumen que la actualización (o los pesos, o los gradientes) cabe en un único subespacio de dimensión $r$, lo que fuerza el compromiso expresividad/parámetros. AdaMSS segmenta las columnas de $W_0$ en $K$ subespacios pequeños e independientes y adapta cada uno con un rango mínimo: representación más compacta, una cota de complejidad gaussiana menor (Teorema 1, mejor garantía de generalización) y la posibilidad de apagar los subespacios irrelevantes para la tarea.
2. **Matemática / Diseño:** *¿Qué papel tiene cada matriz en $W = W_0 + ABC$ y por qué se inicializan así?*
   * **Respuesta:** $A$ sale de la SVD de cada bloque y se congela: marca "hacia dónde" (las direcciones del subespacio). $C$ almacena "cuánto" corregir y empieza a 0, para que al principio el modelo se comporte exactamente como el preentrenado. $B$ es la bisagra que conecta $C$ con $A$; se inicializa ortogonal (QR) para preservar la estructura de $A$ desde el primer paso y que el entrenamiento no se desoriente.
3. **Crítica / Trade-offs:** *¿Por qué el checkpoint de AdaMSS pesa casi como el de LoRA con 13 veces menos parámetros, y por qué no merece la pena `use_asa=True` en este benchmark?*
   * **Respuesta:** Porque guarda en fp32 las proyecciones congeladas de la SVD (~117 MB); lo entrenable son ~11 MB. Con ASA, el default `final_warmup=1000` ni siquiera termina en 750 pasos, y con `600` la mejora (0.6823 frente a 0.6790) es mínima y con una sola semilla y 4 imágenes de test puede ser ruido. Además, la poda congela subespacios durante el entrenamiento, pero el número de parámetros entrenables se cuenta al inicio (2,905,054 en ambos casos).

**Flashcards**
1. **P:** ¿Cuál es la actualización de AdaMSS y qué hace cada matriz?
   **R:** $W^{(k)} = W_0^{(k)} + A^{(k)} B^{(k)} C^{(k)}$ por subespacio: $A$ (SVD del bloque, congelada) da la dirección, $B$ (ortogonal, entrenable) conecta, y $C$ (a 0, entrenable) da la magnitud del cambio.
2. **P:** ¿Qué diferencia hay entre `use_asa=False` y `use_asa=True`?
   **R:** `False` entrena los $K$ bloques todo el tiempo (LoRA con $K$ bloques estructurados). `True` puntúa los bloques cada `mask_interval` pasos y congela los peores hasta quedarse con `asa_target_subspaces` (contados entre todas las capas).
3. **P:** ¿Cómo encuentra AdaMSS los subespacios de $W_0$?
   **R:** Con la SVD truncada calcula $Z^\star = VV^\top$, que sale casi diagonal por bloques; agrupa en clusters las columnas que son combinaciones lineales unas de otras, las reordena para que queden juntas y manda el ruido a una matriz de residuos congelada.
4. **P:** ¿Cómo quedó AdaMSS frente a LoRA $r=32$ en `FLUX.2-klein`?
   **R:** Prácticamente igual (DINOv2 0.6790 frente a 0.6815, con menos drift) usando 2.9M parámetros (0.075%) frente a 38.3M; en la L40S lo supera ligeramente (0.7029 frente a 0.6972).
