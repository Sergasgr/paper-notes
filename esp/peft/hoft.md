# HOFT: Householder Orthogonal Fine-tuning (2025, MLLP/VRAIN - UPV — Moreno Arcas et al.)

**Pregunta con la que lo leí:** ¿Qué limita a OFT, BOFT y HRA, y qué resuelve HOFT con dos matrices ortogonales de Householder?

**Problema:** Los métodos PEFT aditivos como LoRA ($W' = W + BA$) asumen una actualización de rango bajo y alteran la geometría de los pesos preentrenados, mientras que los métodos ortogonales previos (OFT, BOFT, HRA) construyen **una única matriz ortogonal** ($W' = QW$) que **deja congelado el espacio de entrada** ($V^\top$) —limitando su expresividad— y sufren altos costes de tiempo y memoria por invertir bloques o aplicar reflexiones secuenciales.

**Idea clave:** A diferencia del resto de métodos PEFT ortogonales que solo rotan la salida y congelan la entrada, HOFT entrena **dos matrices ortogonales** ($Q_U$ y $Q_V$) construidas en paralelo mediante "espejos" de Householder para multiplicar los pesos preentrenados por delante y por detrás, logrando reorientar tanto la entrada como la salida sin destruir la estructura geométrica aprendida en el preentrenamiento. 

**Método:**
A diferencia de los métodos aditivos, el ajuste fino ortogonal es **multiplicativo**: preserva longitudes, ángulos y la **energía hiperesférica** ($HE(W) = \sum_{i \neq j} \|w_i - w_j\|^{-1}$), manteniendo separadas y diversas las neuronas ya entrenadas.

Visto desde la descomposición **SVD** ($M = U \Sigma V^\top$, donde $V^\top$ rota la entrada, $\Sigma$ estira según importancia y $U$ rota la salida), conservar la energía hiperesférica exige mantener intactos los estiramientos ($\hat{\Sigma} = \Sigma$), pero dejando libres **ambas** rotaciones ($\hat{U}$ y $\hat{V}^\top$). 

Métodos como **OFT, BOFT y HRA** solo aplican una transformación por la izquierda ($\hat{M} = QM = QU\Sigma V^\top$), lo que permite adaptar $U$ pero deja $V^\top$ **congelada** e impide alcanzar todas las matrices posibles (incurriendo en un error acotado por el Orthogonal Procrustes Problem: $\min_{Q \in O(m)} \Vert{}\hat{M} - QM\Vert{}_F \le 2\sqrt{m}\Vert{}M\Vert{}_F$). HOFT resuelve esto entrenando **dos matrices ortogonales**, $Q_U \in O(m)$ y $Q_V \in O(n)$, que actualizan implícitamente tanto $U$ como $V^\top$ sin necesidad de calcular la SVD:

$$
\hat{M} = Q_U M Q_V = \underbrace{(Q_U U)}_{\hat{U}} \Sigma \underbrace{(V^\top Q_V)}_{\hat{V}^\top}
$$

* **Construcción paralela (Transformada CWY + Serie de Neumann):** Cada una de las dos matrices ortogonales acumula $r$ reflexiones de Householder ($H_i = I - \frac{u_i u_i^\top}{\tau_i}$) agrupando los vectores normalizadores como columnas de $U \in \mathbb{R}^{m \times r}$ y $V \in \mathbb{R}^{n \times r}$. Para evitar el producto secuencial de HRA y el coste de invertir matrices densas durante el entrenamiento, se usa la transformada CWY y se separa la matriz triangular superior $S = D + A$ (diagonal + triangular superior estricta), aproximando $S^{-1}$ con los dos primeros términos de la serie de Neumann:
  $$Q_U = I - U S^{-1} U^\top \approx I + U(D^{-1}AD^{-1} - D^{-1})U^\top$$
* **Inicialización aleatoria:** A diferencia de OFT/BOFT (que exigen inicializar a cero), HOFT inicializa parejas consecutivas de vectores gaussianos idénticos $U = (u_1 | u_1 | \dots | u_k | u_k)$. Como reflejar dos veces devuelve el vector original ($H_i H_i = I$), $Q_U = I$ y $Q_V = I$ al inicio del entrenamiento.
* **Variante SHOFT (inspirada en DoRA):** Introduce un vector entrenable de escalado $m \in \mathbb{R}^m$ (inicializado a unos) justo entre ambas rotaciones para ajustar tanto dirección como magnitud, relajando la conservación estricta de energía hiperesférica:
  $$\hat{M} = Q_U m M Q_V$$

**Resultado clave:**
* **Razonamiento de sentido común (8 benchmarks, $r=16$):** **HOFT alcanza 87.4%** en LLaMA3.1-8B (vs. 87.0% LoRA, 87.1% DoRA, 86.5% OFT) y **92.4%** en Qwen2.5-14B (vs. 92.1% LoRA, 92.0% DoRA); **SHOFT logra 92.2%** en Phi4-14B (vs. 91.8% DoRA, 91.7% LoRA).
* **Razonamiento matemático (MetaMathQA sobre LLaMA2-7B, $r=8$):** **HOFT obtiene 56.6%** en GSM8K (vs. 50.2% LoRA, 53.5% DoRA, 56.3% HRA) y **SHOFT logra 9.8%** en MATH (vs. 7.8% LoRA, 9.1% DoRA, 9.3% HRA).
* **Modelos cuantizados a 4 bits (QHOFT / QSHOFT):** En Qwen2.5-32B (0.07% parámetros), **QSHOFT sube a 75.0%** en GSM8K (vs. 68.0% QLoRA, 68.4% QDoRA) y **QHOFT alcanza 80.6%** en Orca-Math (vs. 77.4% QLoRA, 61.4% QDoRA).
* **Generación personalizada (DreamBooth en SD 1.5) y velocidad:** $\text{HOFT}_{r=4}$ logra **0.680 DINO / 0.810 CLIP-I** (superando a LoRA 0.613/0.765 y DoRA 0.633/0.781), entrenando un **732.5% más rápido que HRA** (4.3h vs. 35.8h) y un **72.1% más rápido que OFT** (4.3h vs. 7.4h).

**Limitaciones:**
* **Qué no prueban:** No evalúan modelos multimodales (*visual instruction tuning*) ni estrategias de compresión de parámetros mediante bancos de vectores (tipo VB-LoRA).
* **Supuestos:** La aproximación de Neumann para $S^{-1}$ requiere que el rango sea mucho menor que la dimensión de la matriz ($r \ll m$). En matrices de pesos de baja dimensión, HOFT y SHOFT no logran imponer la ortogonalidad completa y el error de energía hiperesférica crece rápidamente con $r$.
* **Coste:** Aunque es mucho más rápido que HRA, OFT y DoRA, sigue siendo ~33–35% más lento que LoRA y consume bastante más pico de VRAM que LoRA (especialmente en capas fusionadas QKV/Up-Gate como en Phi4-14B: 78.6 GB vs. 49.9 GB de LoRA).

**Conexión:** `#peft` `#orthogonal-finetuning` `#svd` `#householder` `#quantization` (alternativa directa a LoRA/DoRA cuando importa preservar el conocimiento geométrico base o adaptar modelos cuantizados a 4 bits sin degradar generalización).

**Preguntas**
1. **Conceptual:** *¿Por qué OFT, BOFT y HRA tienen menor expresividad teórica que HOFT aunque todos usen transformaciones ortogonales?*
   * **Respuesta:** Porque multiplican los pesos $M = U\Sigma V^\top$ por un solo lado ($\hat{M} = QM$), lo que solo permite reorientar el espacio de salida ($U$) dejando congelado el de entrada ($V^\top$). HOFT usa dos matrices ($\hat{M} = Q_U M Q_V$) para transformar ambos espacios manteniendo $\Sigma$ intacta.
2. **Matemática / Diseño:** *¿Cómo consigue HOFT inicializar sus matrices ortogonales de forma aleatoria asegurando que al principio del entrenamiento no alteren el modelo base ($Q_U = Q_V = I$)?*
   * **Respuesta:** Emparejando vectores de Householder consecutivos idénticos ($u_1 | u_1 | \dots | u_k | u_k$) muestreados de una distribución gaussiana. Como una reflexión de Householder es involutiva ($H_i^2 = I$), cada par se cancela a sí mismo en el producto inicial, pero durante el entrenamiento sus gradientes divergen libremente.
3. **Crítica / Trade-offs:** *¿Cuándo falla la aproximación rápida de HOFT y qué ventaja práctica ofrece frente al regularizador L2 (weight decay)?*
   * **Respuesta:** Falla cuando la matriz de pesos tiene dimensión pequeña o $r$ es grande respecto a $m$, porque truncar la serie de Neumann en dos términos acumula error de ortogonalidad. Como ventaja, la aproximación $I + U(D^{-1}AD^{-1} - D^{-1})U^\top$ es invariante al escalado uniforme de $U$, por lo que es inmune al *weight decay*.

**Flashcards**
1. **P:** ¿Cuál es la diferencia fundamental entre lo que hace LoRA y lo que hace HOFT sobre una matriz de pesos preentrenada $M = U\Sigma V^\top$?
   **R:** LoRA es **aditivo** ($M + BA$, asume actualización de rango bajo y altera la geometría de los pesos). HOFT es **multiplicativo** ($Q_U M Q_V$, rota/refleja los espacios de entrada $V^\top$ y salida $U$ dejando intactos los valores singulares $\Sigma$, lo que preserva la energía hiperesférica del modelo preentrenado).
2. **P:** ¿Qué problema computacional resuelve la combinación de transformada CWY + serie de Neumann en HOFT?
   **R:** Evita aplicar las $r$ reflexiones de Householder de forma secuencial (el cuello de botella de HRA) y elimina la necesidad de invertir la matriz triangular $S \in \mathbb{R}^{r \times r}$ en cada paso de entrenamiento, aproximando $S^{-1} \approx D^{-1} - D^{-1}AD^{-1}$ mediante la inversa trivial de una matriz diagonal $D$.
3. **P:** ¿En qué se diferencia la $r$ de HOFT respecto a la $r$ de HRA en número de parámetros?
   **R:** HRA usa $r$ vectores de Householder para una sola matriz ($rm$ parámetros), mientras que HOFT usa $r$ vectores para $Q_U$ y otros $r$ para $Q_V$, sumando $r(m+n)$ parámetros (el doble en matrices cuadradas).
4. **P:** ¿Qué añade SHOFT sobre HOFT y qué propiedad teórica sacrifica a cambio?
   **R:** Añade un vector entrenable de escalado $m$ entre ambas matrices ortogonales ($\hat{M} = Q_U m M Q_V$, inspirado en DoRA) para ajustar magnitudes además de direcciones; a cambio, deja de garantizar la conservación estricta de la energía hiperesférica.