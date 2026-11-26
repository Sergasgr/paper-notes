# VeLoRA: Memory Efficient Training using Rank-1 Sub-Token Projections (NeurIPS 2024, Huawei Noah's Ark Lab — Miles et al.)

**Pregunta con la que lo leí:** ¿Cómo reduce VeLoRA la memoria de activaciones comprimiendo la copia de $x$ que se guarda para el *backward*, y por qué apenas ahorra memoria en el benchmark de `FLUX.2-klein`?

**Problema:** En la operación $y = Wx + b$ de una capa, lo que colapsa la memoria es la necesidad de guardar esa $x$ (las entradas que recibe la capa), porque la regla de la cadena exige $\partial \mathcal{L}/\partial W = (\partial \mathcal{L}/\partial y) \cdot x$; los métodos PEFT reducen los pesos entrenables y los estados del optimizador, pero siguen guardando las activaciones completas, y las alternativas (gradient checkpointing, GaLore) pagan con recomputación o SVD periódicas.

**Idea clave:** Durante el *forward*, en lugar de guardar las activaciones completas, dividir cada token en sub-tokens y comprimirlos proyectándolos sobre un único vector fijo (rango 1); en el *backward*, reconstruirlos al vuelo de forma tosca para calcular los gradientes. Lo que pasa a la siguiente capa no se comprime: solo la "fotocopia" guardada para el gradiente.

**Método:**
VeLoRA no es un adaptador nuevo, sino una modificación del *backprop*: comprime los vectores de activación intermedios que el paper llama tensor $Z$ (la $x$ de $y = Wx$) y los reconstruye solo cuando se necesitan. Con LoRA, $W$ está congelada y la $x$ se guarda para calcular el gradiente de la matriz $A$ de LoRA: esa es la copia que comprime VeLoRA en `peft`.

```math
Z \xrightarrow{\text{group}} z \in \mathbb{R}^{B \times \frac{ND}{M} \times M} \xrightarrow{\;z\,v\;} z_p \in \mathbb{R}^{B \times \frac{ND}{M} \times 1}, \qquad \hat{z} = z_p\, v^\top \;\;\Rightarrow\;\; \text{proj}_v(z) = (z\,v)\,v^\top
```

* **Agrupación:** cada token de dimensión $D$ se divide en sub-tokens más pequeños de tamaño $M \ll D$. En `peft` lo controla `num_groups`: con `num_groups=32`, cada token de 3,072 valores pasa a guardarse como 32 números.
* **Proyección de rango 1:** cada sub-token se proyecta sobre un subespacio unidimensional fijo, un vector unitario $v$ que se calcula con la media del primer batch y se congela (sin SVD, a diferencia de GaLore). En memoria solo queda $z_p$.
* **Reconstrucción y por qué converge:** en el *backward*, cada sub-token se reconstruye al vuelo como $\hat{z} = z_p v^\top$ y se usa en lugar de $x$ para el gradiente de los pesos. La reconstrucción es tosca y pierde propiedades del gradiente original como la magnitud y la dirección, pero preserva localmente la similitud entre gradientes, lo que permite que el modelo converja con éxito.

**Resultado clave:**
* **LLaMA + QLoRA (Alpaca → MMLU 5-shot):** QLoRA + VeLoRA en LLaMA-7B saca **39.5%** con **4.88 GB** frente a QLoRA (39.0% con 5.77 GB, **−15.4%** de memoria); en LLaMA-13B, 48.0% con 8.48 GB frente a 47.5% con 9.91 GB (−14.4%).
* **RoBERTa-base (GLUE):** **2.23 GB** frente a LoRA (2.71 GB, −18%), GaLore (4.04 GB, −45%) y *full fine-tuning* (4.64 GB), perdiendo solo 0.37 puntos frente a este último.
* **Frente a gradient checkpointing (LLaMA-7B, Alpaca):** un 30% más rápido (1.50 frente a 1.14 it/s), aunque checkpointing ocupa algo menos (7.31 GB frente a 8.35 GB).
* **Mis experimentos (`FLUX.2-klein`, LoRA $r=32$ + VeLoRA en los 11 módulos):** **DINOv2 = 0.6907** / drift 0.2588 frente a LoRA (0.6815 / 0.2721), pero con solo **−34 MB** de memoria (10,332 MB frente a 10,366 MB) y un tiempo parecido (1,129 s frente a 1,161 s), con los mismos 38,338,560 parámetros entrenables.

**Limitaciones:**
* **Qué no prueban:** solo Transformers (no CNN, RNN ni SSM, y no está claro cómo extenderlo). Los propios autores reconocen que su ventaja es la memoria: el tiempo de entrenamiento sigue siendo un problema.
* **Supuestos:** el ahorro depende de que las activaciones guardadas dominen la memoria. En el benchmark, los parámetros por defecto usan gradient checkpointing (`use_gc: true`), que ya resuelve el mismo problema por otra vía (no guarda las activaciones de cada bloque y las recomputa en el *backward*), así que a VeLoRA apenas le quedan activaciones que comprimir. Aun sin checkpointing, el margen sería limitado: unos 7.2 GiB de los ~10 GiB son los pesos del modelo base en bf16, y VeLoRA solo comprime las entradas guardadas para `lora_A`.
* **Coste (el argumento de "usar un rango mayor"):** la run con $r=64$ duplica los parámetros entrenables pero usa 718 MB más sin mejorar la calidad (0.6886): el ahorro de VeLoRA es fijo, mientras que el coste de LoRA (parámetros, gradientes y estados de Adam) crece con $r$. Y la run de solo Value+Down (0.6568 con 9,396 MB) no aísla VeLoRA: en `peft` se aplica a todas las capas LoRA, así que replicar la ubicación del paper obligó a reducir `target_modules` de 11 a 3, y casi toda su reducción (−936 MB) viene de adaptar menos capas.

**Conexión:** `#peft` `#memory-efficient` `#activations` `#gradient-checkpointing` `#lora` `#flux2-klein` (no cambia qué se entrena sino qué se guarda para el *backward*: combinable con LoRA y QLoRA, complementario a GaLore, que comprime gradientes, y competidor de gradient checkpointing, que recomputa; como el adaptador sigue siendo LoRA $r=32$, su +1.4% de DINOv2 con una semilla y 4 imágenes de test puede ser ruido).

**Preguntas**
1. **Conceptual:** *¿Qué comprime exactamente VeLoRA y qué deja intacto?*
   * **Respuesta:** Solo la copia de la entrada $x$ (el tensor $Z$) que se guarda durante el *forward* para calcular después $\partial \mathcal{L}/\partial W = (\partial \mathcal{L}/\partial y) \cdot x$; en `peft`, la entrada guardada para `lora_A`. El resultado del *forward* que pasa a la siguiente capa no se comprime, ni los pesos ni los estados del optimizador.
2. **Matemática / Diseño:** *¿Cómo funcionan la compresión y la reconstrucción, y por qué basta con un único vector fijo?*
   * **Respuesta:** Se reagrupa cada token en sub-tokens de tamaño $M$, se guarda $`z_p = z\,v`$ (un número por sub-token) y en el *backward* se reconstruye $\hat{z} = z_p v^\top$. Al trabajar con sub-tokens pequeños, un subespacio de dimensión 1 fijo (la media del primer batch) es suficiente: la reconstrucción pierde magnitud y dirección del gradiente, pero conserva localmente la similitud entre gradientes, que es lo que necesita la optimización para converger. Además evita SVD y recomputación.
3. **Crítica / Trade-offs:** *¿Por qué en `FLUX.2-klein` VeLoRA solo ahorra 34 MB, y por qué la run con $r=64$ contradice el argumento del paper de que permite "usar un rango mayor"?*
   * **Respuesta:** Porque el benchmark ya usa gradient checkpointing, que no guarda las activaciones de cada bloque, y porque la mayor parte de la VRAM (~7.2 GiB) son los pesos base en bf16. Con un ahorro casi nulo y fijo, no hay nada que compense la memoria extra de duplicar $r$ (parámetros, gradientes y estados de Adam de LoRA): $r=64$ usó 718 MB más sin mejorar el DINOv2.

**Flashcards**
1. **P:** ¿Por qué hay que guardar la entrada $x$ de cada capa durante el entrenamiento?
   **R:** Porque el gradiente de los pesos es $\partial \mathcal{L}/\partial W = (\partial \mathcal{L}/\partial y) \cdot x$: sin $x$ no hay forma de actualizar $W$ (o `lora_A` en LoRA).
2. **P:** ¿Cuáles son los dos pasos de compresión de VeLoRA?
   **R:** Agrupación (dividir cada token en sub-tokens pequeños) y proyección de rango 1 sobre un vector fijo $v$ (media del primer batch, congelado). En el *backward* se reconstruye con $z_p v^\top$.
3. **P:** ¿En qué se diferencia VeLoRA de gradient checkpointing?
   **R:** Checkpointing no guarda las activaciones y las recomputa en el *backward* (más cómputo). VeLoRA las guarda comprimidas y las reconstruye de forma tosca, sin recomputar (más rápido: un 30% en LLaMA-7B).
4. **P:** ¿Por qué VeLoRA apenas ahorró memoria en `FLUX.2-klein` (−34 MB)?
   **R:** Porque el benchmark usa `use_gc: true` (gradient checkpointing), así que casi no quedan activaciones guardadas que comprimir, y la mayor parte de la VRAM son los pesos base en bf16.
