# Super-Tuning: From Activation-Aware Pruning to Sparse Fine-Tuning (2026, KAUST — Ilin et al.)

**Pregunta con la que lo leí:** ¿Cómo elige Super-Tuning qué pesos entrenar sin calcular gradientes, y por qué en `FLUX.2-klein` me funcionó mejor TopK que el BottomK que recomienda el paper?

**Problema:** El *sparse tuning* (actualizar solo una fracción minúscula de los pesos originales, manteniendo la inmensa mayoría congelada) tradicionalmente requiere cálculos pesados de gradientes o varias fases de entrenamiento para encontrar qué parámetros modificar, tanto que a veces anula la eficiencia que se intentaba conseguir.

**Idea clave:** Reutilizar una métrica de poda (Wanda: magnitud del peso × norma de la activación que le llega) para elegir, con un único *forward* de calibración y sin gradientes, una máscara fija de pesos a entrenar (**Super**), y opcionalmente sumarle un LoRA dentro del mismo presupuesto de parámetros (**Supra**).

**Método:**
A diferencia de LoRA, que aprende desde cero dos matrices pequeñas cuyo producto se suma como un parche, en *sparse tuning* la matriz de cambios que se suma a los pesos originales está llena de ceros, salvo unos pocos pesos específicos que sí permitimos entrenar:

$$
\hat{W} = W + M \odot U \quad (\text{Super}), \qquad \hat{W} = W + M \odot U + \gamma\, L R \quad (\text{Supra}), \qquad A_{ij} = |W_{ij}| \cdot \|X_j\|_2
$$

* **Puntuación Wanda:** $|W_{ij}| \cdot \|X_j\|_2$, la magnitud del peso multiplicada por la norma de la activación de entrada que le llega. Si la $X$ que entra es enorme, el impacto de ese peso en la salida es gigantesco; si la $X$ es 0, el peso es inútil en ese momento. Se calcula con un único *forward pass* de calibración (cero cálculo de gradientes).
* **Máscara y actualización:** con esa puntuación se construye una máscara $M$ del mismo tamaño que $W$, con un 1 en los $K$ pesos seleccionados (TopK: los de mayor puntuación; BottomK: los de menor) y un 0 en el resto. $U$ se inicializa llena de ceros y, al multiplicarla por la máscara, solo sobreviven las actualizaciones de los pesos seleccionados. El paper encuentra que BottomK suele dar los mejores resultados, porque altera las partes menos críticas del modelo, aunque la mejor dirección depende del modelo y del entrenamiento.
* **Supra (híbrido):** suma a Super la actualización estándar de LoRA ($\gamma$ fija la fuerza del parche de bajo rango): actualizaciones milimétricas en ciertos pesos más un ajuste global de la capa. A diferencia de híbridos previos como RoSA, la máscara se elige una sola vez sin entrenamiento y bajo un límite estricto: $\lambda$ reparte el presupuesto entre el tamaño de LoRA y los pesos de Super, para no entrenar más parámetros que un LoRA tradicional.
* **En `peft`:** la selección es por magnitud $|W|$, no por Wanda, así que no hay *forward* de calibración (según la documentación de `peft`, es la mejor configuración del paper). $U$ no existe como matriz completa: solo se guardan y entrenan los $K$ valores seleccionados con sus posiciones. No hay $\lambda$: el reparto de Supra se hace eligiendo `sparsity` y `r`.

**Resultado clave:**
* **Meta-Llama-3-8B (Math17K → 6 benchmarks aritméticos, presupuesto de LoRA $r=8$):** Supra-Mag (BottomK por magnitud + LoRA, $\lambda=0.3$) logra **79.12%** y Supra con Wanda 78.66%, frente a LoRA (**73.17%**), RoSA (75.16%) y *full fine-tuning* (74.87%). Pero Super puro (Wanda, BottomK) se queda en 72.73%, por debajo de LoRA, y la máscara solo por magnitud (BottomK) ya da 79.02%.
* **Llama-3.2-1B:** el mejor es Supra ($\lambda=0.8$) con **62.23%** frente a 61.07% de LoRA; Super puro, 55.46%.
* **Mis experimentos (`FLUX.2-klein`, TopK, densidad 1.31% en los 11 tipos de módulo para igualar a LoRA $r=32$, lr `3e-4`):** **DINOv2 = 0.7108** / drift 0.2709 frente a LoRA (0.6815 / 0.2721), con 38,324,370 parámetros, +4 MiB de memoria y un 4.8% menos de tiempo (1,105 s). Con lr `1e-4`: TopK 0.7048, BottomK 0.6938 y Supra (BottomK al 0.94% + LoRA $r=9$) 0.6894.

**Limitaciones:**
* **Qué no prueban:** una sola semilla, tablas con el mejor calendario (1 o 3 épocas) de cada método y solo razonamiento aritmético con dos modelos Llama (1B y 8B). Nada de difusión ni de tareas abiertas (resumen, diálogo, código).
* **Supuestos:** Wanda es solo un proxy de importancia, y la dirección TopK/BottomK depende del modelo: en el paper suele ganar BottomK, pero en mis pruebas TopK da mejor DINOv2 en test. La máscara es fija: un peso descartado al principio no puede activarse después. Es sensible al lr: BottomK con `1e-3` rompe el modelo (DINOv2 0.5263, drift 0.423), igual que TinyLoRA con `1e-2`. Con una densidad del 0.1% (2.9M parámetros) cae a 0.4691.
* **Coste:** el checkpoint pesa el doble que el de LoRA (292.4 MiB), porque por cada peso entrenable se guarda su valor (fp32, 4 bytes) y su posición (int32, 4 bytes). Por ese mismo almacenamiento compacto, la memoria de entrenamiento apenas sube (+4 MiB). El paper mide una implementación sin kernels dispersos optimizados.

**Conexión:** `#peft` `#sparse-tuning` `#pruning` `#wanda` `#lora` `#flux2-klein` `#tweft` (lleva la poda a PEFT: en vez de qué pesos quitar, qué pesos entrenar. Como BEFT y TWEFT, elige qué entrenar antes del *fine-tuning*, pero a nivel de peso individual y sin warmup. Bossan también destacó que da muy buenos resultados, superando a LoRA).

**Preguntas**
1. **Conceptual:** *¿Qué diferencia el sparse tuning de LoRA, y qué problema de los métodos dispersos anteriores resuelve Super?*
   * **Respuesta:** LoRA aprende desde cero un parche de bajo rango que se suma a toda la matriz; el *sparse tuning* modifica directamente unos pocos pesos originales y deja el resto exactamente igual. Los métodos dispersos previos (SIFT, RoSA) eligen esos pesos con gradientes o varias fases de entrenamiento; Super los elige con una métrica de poda en un solo *forward* de calibración, sin gradientes.
2. **Matemática / Diseño:** *¿Cómo garantiza $M \odot U$ que solo cambien los pesos elegidos, y por qué el paper prefiere BottomK?*
   * **Respuesta:** $M$ tiene un 1 solo en los $K$ pesos seleccionados: cualquier actualización de $U$ fuera de la máscara se multiplica por 0 y no llega al modelo, y como $U$ empieza a 0, al principio $\hat{W} = W$. BottomK elige los pesos menos influyentes según Wanda: modificarlos tiene menos riesgo de romper comportamientos críticos que el modelo ya aprendió.
3. **Crítica / Trade-offs:** *Si en el paper Super puro queda por debajo de LoRA y la máscara solo por magnitud iguala a Wanda, ¿qué aporta realmente el método?*
   * **Respuesta:** Lo que funciona es la máscara fija elegida sin entrenamiento combinada con LoRA (Supra), con el mismo presupuesto de parámetros; el peso de la activación no parece imprescindible (79.02% solo con magnitud frente a 79.12%), que es justo lo que usa `peft`. En `FLUX.2-klein`, Super puro con TopK sí superó a LoRA, pero la dirección depende del modelo y las diferencias de ~0.01 con una semilla y 4 imágenes de test pueden ser ruido.

**Flashcards**
1. **P:** ¿Qué es la puntuación Wanda y cómo la usa Super?
   **R:** $|W_{ij}| \cdot \|X_j\|_2$ (magnitud del peso × norma de su activación de entrada). Super la calcula con un *forward* de calibración y entrena solo los $K$ pesos con menor (BottomK) o mayor (TopK) puntuación.
2. **P:** ¿Cuál es la diferencia entre Super y Supra?
   **R:** Super: $\hat{W} = W + M \odot U$ (solo pesos dispersos). Supra: $\hat{W} = W + M \odot U + \gamma LR$ (además un LoRA), con $\lambda$ repartiendo el presupuesto para no superar los parámetros de un LoRA.
3. **P:** ¿Por qué el checkpoint de Super-Tuning pesa el doble que el de LoRA con los mismos parámetros?
   **R:** Porque por cada peso entrenable guarda su valor (fp32, 4 bytes) y su posición (int32, 4 bytes).
4. **P:** ¿Qué cambia la implementación de `peft` respecto al paper?
   **R:** Selecciona por magnitud $|W|$ (sin *forward* de calibración), solo guarda los $K$ valores y sus posiciones (no la $U$ completa) y no tiene $\lambda$: el reparto de Supra se elige con `sparsity` y `r`.
