# Robust and Efficient Fine-tuning of LLMs with Bayesian Reparameterization of Low-Rank Adaptation — MonteCLoRA (2025, IIT Delhi / JPMorgan AI Research — Seth et al.)

**Pregunta con la que lo leí:** ¿Cómo usa MonteCLoRA un promedio Monte Carlo de pesos con ruido para que LoRA sea menos sensible al *learning rate*, y por qué no superó a LoRA en `FLUX.2-klein`?

**Problema:** LoRA es muy sensible a hiperparámetros como el *learning rate* o el tamaño del lote: si no se ajustan bien, el modelo se vuelve inestable, necesita demasiadas iteraciones o directamente diverge, y afinarlos por fuerza bruta (*grid search*) es computacionalmente inviable en LLMs.

**Idea clave:** En lugar de buscar un único valor estático y perfecto para los pesos de adaptación, MonteCLoRA aprende una distribución a posteriori para la matriz $A$ de LoRA: en cada paso genera varios candidatos ($A$ más ruido), los promedia con pesos de Dirichlet y controla el tamaño del ruido con una Wishart, empujando al modelo hacia una **cuenca de óptimos amplia y plana**.

**Método:**
MonteCLoRA es una alternativa bayesiana a LoRA. Mediante una estimación Monte Carlo promedia varios pesos posibles (aproxima la marginalización de la distribución predictiva), de modo que el modelo promedia la incertidumbre y no se desestabiliza por una configuración subóptima:

$$
\Sigma \sim \mathcal{W}(V, n_{\text{in}}), \quad \pi \sim \text{Dir}(\alpha), \quad S^{(k)} \sim \mathcal{N}(0, \Sigma), \qquad A_{\text{MonteCLoRA}} = \mu + \epsilon \sum_{k=1}^{N} \pi_k\, S^{(k)}
$$

* **Factorización y parámetros extra:** aprender la distribución conjunta de toda $A$ es inviable, así que se asume que cada columna se comporta como una distribución independiente. Se entrenan $\mu$ (los pesos de LoRA), la escala $V$ de la Wishart (diagonal) y la concentración $\alpha$ de la Dirichlet: solo $\mathcal{O}(r)$ parámetros extra por capa.
* **Candidatos (Wishart) y pesos (Dirichlet):** se generan $N$ candidatos posibles. La Wishart genera la matriz de covarianza, es decir, cuánto pueden alejarse los candidatos de $\mu$, escalada por $\epsilon$ (`sample_scaler`); la Dirichlet reparte el 100% de importancia entre los $N$ candidatos. Ejemplo con un solo peso: $\mu = 10.0$, ruidos $+1.0$, $-0.5$, $+0.2$ con pesos 0.6, 0.3 y 0.1 → $+0.47$; con $\epsilon = 0.1$, el peso usado es 10.047. El *backprop* actualiza $\mu$ y, a la vez, los parámetros de Wishart y Dirichlet.
* **Por qué da robustez:** al inyectar este ruido durante el entrenamiento, el valor del parámetro "vibra" en cada iteración, así que el modelo evita quedarse en mínimos estrechos o afilados (agujeros frágiles) y se va a una cuenca amplia y plana (*basin of optima*), donde pequeños errores en el lr ya no lo desestabilizan. Se añaden una pérdida KL con los priors y una *cooperative loss*, que obliga a que los $N$ candidatos participen por igual (es mínima cuando todos los pesos de Dirichlet valen $1/N$). En inferencia y al hacer *merge* no se muestrea ruido: se usa la $A$ media, como un LoRA normal.

**Resultado clave:**
* **NLU (RoBERTa-base, GLUE y SuperGLUE):** +0.5% de accuracy y +1.6% de robustez frente a LoRA sin regularizar.
* **Sentido común (LLaMA-1-7B, varias configuraciones de lr y batch):** la dispersión de accuracy baja un **53%** (2.19 frente a 4.69 puntos de LoRA), con una media a la par (71.1% frente a 71.3%). En LLaMA-3.2-3B-Instruct, **−62%** de dispersión y +13.3% de robustez.
* **Mis experimentos (`FLUX.2-klein`, $r=16$ y $N=2$ por los 16 GB de VRAM):** con $\epsilon$=`5e-3` y *cooperative loss*: **DINOv2 = 0.6916** / drift 0.2846 (19,170,720 parámetros, 12,974 MB, 1,284 s); con $\epsilon$=`2.5e-3`: 0.6784 / 0.2690; sin *cooperative loss*: 0.6780 / 0.2615. En una RTX 5090 probé la config del paper ($r=32$, $N=4$: 0.6453, 22,018 MB) y $r=32$ con $N=2$ (0.6831, 16,402 MB): ninguna mejoró la entrada actual ni a LoRA $r=32$ en la misma GPU (0.6944 / drift 0.2439, 10,342 MB), así que la entrada se mantiene y esas dos runs van al graveyard.

**Limitaciones:**
* **Qué no prueban:** modelos de lenguaje de hasta 7B (sin difusión). Su ganancia es la **robustez** (menos dispersión entre configuraciones), no una media más alta: en LLaMA-1-7B la media queda igual que LoRA.
* **Supuestos:** reduce la sensibilidad al lr, pero no es que funcione mágicamente con cualquier valor absurdo: da más margen de error. Y añade otro hiperparámetro sensible, $\epsilon$: según el paper, un $\epsilon$ alto puede hacer diverger el modelo y uno bajo, sobreajustar (0.005 fue el mejor), además del peso de la KL. El otro $\epsilon$ del paper es $2.5 \times 10^{-5}$ (para LLaMA-3.2), no `2.5e-3`: mi `2.5e-3` es simplemente la mitad de `5e-3`. Mi entrada con $r=16$ y $N=2$ tampoco era una comparación justa con LoRA $r=32$.
* **Coste:** el muestreo Monte Carlo añade tiempo de entrenamiento (1.23–1.30x LoRA en LLaMA-3.1-8B, incluso con muestras precalculadas en un *buffer*). En `FLUX.2-klein` la memoria extra viene del *buffer* de ruido que MonteCLoRA genera por adelantado, que crece con `buffer_size` × $N$ × $r$: 12,974 MB con $r=16$/$N=2$ y 22,018 MB con $r=32$/$N=4$, frente a ~10.4 GB de LoRA.

**Conexión:** `#peft` `#bayesian` `#monte-carlo` `#robustness` `#lora` `#learning-rate` `#flux2-klein` (otra respuesta a la sensibilidad al lr que vi en HiRA, LoHa y FourierFT, pero regularizando en vez de buscar el lr; en inferencia es un LoRA normal. La entrada y las runs de la 5090 se discutieron en el PR #3806).

**Preguntas**
1. **Conceptual:** *¿Qué problema de LoRA ataca MonteCLoRA y por qué un promedio de pesos con ruido lo mitiga?*
   * **Respuesta:** La sensibilidad a hiperparámetros como el lr, que hace que LoRA sea inestable o diverja si no están bien ajustados. Al entrenar con un ensamble de candidatos ruidosos alrededor de $\mu$, el modelo no puede quedarse en mínimos estrechos (un poco de ruido lo sacaría) y converge a una cuenca amplia y plana, donde una configuración algo peor no lo desestabiliza.
2. **Matemática / Diseño:** *¿Qué papel tienen la Wishart, la Dirichlet, $\epsilon$ y la cooperative loss?*
   * **Respuesta:** La Wishart genera la covarianza del ruido: cuánto pueden alejarse los $N$ candidatos de $\mu$. La Dirichlet genera los pesos de la mezcla (suman 1). $\epsilon$ escala el ruido antes de sumarlo a $\mu$ (demasiado alto diverge, demasiado bajo sobreajusta). La *cooperative loss* evita que un solo candidato acapare toda la importancia, empujando los pesos de Dirichlet hacia $1/N$.
3. **Crítica / Trade-offs:** *¿Por qué MonteCLoRA no superó a LoRA en `FLUX.2-klein` aunque el paper lo vende como más robusto?*
   * **Respuesta:** Porque su ventaja es una menor dispersión al variar los hiperparámetros, no una media más alta (en LLaMA-1-7B la media queda igual que LoRA), y el benchmark mide una sola run con el lr por defecto, que no capta esa ventaja. Además paga en memoria (el *buffer* de ruido: 22 GB con la config del paper) y en tiempo, y añade $\epsilon$ como hiperparámetro sensible.

**Flashcards**
1. **P:** ¿Cómo se calcula la matriz $A$ que usa la red en MonteCLoRA durante el entrenamiento?
   **R:** $A = \mu + \epsilon \sum_{k=1}^{N} \pi_k S^{(k)}$, con $S^{(k)} \sim \mathcal{N}(0, \Sigma)$, $\Sigma$ de una Wishart y $\pi$ de una Dirichlet.
2. **P:** ¿Qué controla la Wishart y qué controla la Dirichlet?
   **R:** La Wishart, la dispersión (tamaño del ruido) de los candidatos; la Dirichlet, el porcentaje de importancia de cada candidato (suman 1).
3. **P:** ¿Qué se usa en inferencia y al hacer *merge*?
   **R:** La $A$ media ($\mu$), sin muestrear ruido: queda como un LoRA normal, sin coste extra.
4. **P:** ¿De dónde viene la memoria extra de MonteCLoRA en `FLUX.2-klein`?
   **R:** Del *buffer* de ruido que genera por adelantado, que crece con `buffer_size` × $N$ × $r$ (22,018 MB con $r=32$, $N=4$).
