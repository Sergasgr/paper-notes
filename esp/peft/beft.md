# BEFT: Bias-Efficient Fine-Tuning of Language Models (2025, Lund University / Google DeepMind — Huang et al.)

**Pregunta con la que lo leí:** ¿Cómo elige BEFT qué familia de biases ($b_q$, $b_k$ o $b_v$) reentrenar, y por qué se queda corto al llevarlo a generación de imágenes con `FLUX.2-klein`?

**Problema:** Las técnicas de PEFT que solo actualizan los sesgos (*biases*), como BitFit, tienen muy buenos resultados en datasets pequeños, casi igualando al *full fine-tuning* (FF), pero no está claro qué bias conviene reentrenar: elegir al azar entre $b_q$, $b_k$ y $b_v$ no garantiza un buen rendimiento, y los criterios previos fallan (la **magnitud** del cambio ignora el ángulo y la **información de Fisher** empírica da un ranking estático que no cambia con la cantidad de datos).

**Idea clave:** Tras un reentrenamiento previo (*warmup*) de todos los biases, BEFT mide con proyecciones geométricas cuánto ha cambiado cada bias **en longitud y en ángulo**, y reentrena solo la "familia" ganadora en todas las capas (casi siempre $b_v$, *value*), con el resto de la red congelada.

**Método:**
A diferencia de otras técnicas de reentrenamiento (como LoRA o Adapters), BEFT no requiere reparametrizaciones auxiliares ni añadir capas nuevas a la arquitectura: solo actualiza biases que el modelo ya tiene. Lo que aporta es cómo seleccionar de forma inteligente **qué** bias reentrenar, dividiendo los de atención en $b_q$, $b_k$ y $b_v$.

En lugar de fijarse solo en si la "flecha" creció o encogió, compara el bias antes ($b_{\mathcal{T}}^{(l),\text{pre}}$) y después ($b_{\mathcal{T}}^{(l),\text{post}}$) del warmup proyectando el vector más corto sobre el más largo (el reentrenado sobre el original, o a la inversa), lo que captura a la vez el cambio de longitud y de ángulo $\alpha$:

```math
\mathcal{I}(b_{\mathcal{T}}) = \frac{1}{L} \sum_{l=1}^{L} \left( 1 - \frac{b_{\mathcal{T}}^{(l),\text{pre}} \cdot b_{\mathcal{T}}^{(l),\text{post}}}{\max\left(\|b_{\mathcal{T}}^{(l),\text{pre}}\|_2^2,\ \|b_{\mathcal{T}}^{(l),\text{post}}\|_2^2\right)} \right), \qquad \mathcal{T}^* = \arg\max_{\mathcal{T} \in \{q,k,v\}} \mathcal{I}(b_{\mathcal{T}})
```

* **Warmup y score:** se reentrenan todos los biases y se compara cómo han variado respecto a los originales. Aquellos cuya proyección se reduce más (han variado mucho en longitud y/o dirección) dan una fracción pequeña y, al restarla a 1, obtienen el score más alto.
* **Selección de la familia:** se promedian los scores en todas las capas para cada tipo de bias y gana una única "familia" completa (toda la de query, la de key o la de value). El fine-tuning real congela el resto de la red y modifica a la vez esa familia en todas las capas. $b_v$ suele ser la ganadora indiscutible, y el paper muestra que el hallazgo se reutiliza en otros datasets y LLMs sin repetir el warmup.
* **En `peft` y en `FLUX.2-klein`:** en `peft` no se hace esta selección: BEFT añade un bias nuevo, inicializado a 0, a cada capa de `target_modules`. Las capas de FLUX.2 no tienen bias (`bias=False`), así que en mi experimento se entrenaron biases nuevos en todas las capas de atención y FF: los 890,880 parámetros son exactamente la suma de los `out_features` de las 80 capas adaptadas, no solo $b_v$.

**Resultado clave:**
* **Selección del bias (BERT-base, GLUE *low-data*):** el bias que elige BEFT ($b_v$) es el mejor en todos los datasets: **64.5%** de media frente a **57.8%** ($b_q$) y **53.5%** ($b_k$). La magnitud elige $b_q$ en SST-2 y CoLA y se equivoca.
* **Eficiencia (BERT-base, RTE *low-data*):** entrenando solo $b_v$ (**0.01%** de parámetros) logra **58.53%** en **132.9 s**, frente a todos los biases (BitFit: 0.09%, 56.40%, 144.9 s) y *full fine-tuning* (100%, 57.46%, 206.1 s). Además, promediar los $b_v$ de tareas distintas (*task arithmetic*) las combina: el merge SST-2 + RTE da 81.3% / 55.5%.
* **LLMs autorregresivos (OPT-6.7B, *low-data*):** con un **0.02‰** de parámetros (30x menos que LoRA), $b_v$ logra **95.2 / 82.6 / 96.4%** en SST-2 / RTE / CB, frente a LoRA (0.6‰: 50.5 / 47.2 / 50.0) y Prefix-Tuning (0.2‰: 95.1 / 78.3 / 89.2).
* **Mis experimentos (`FLUX.2-klein`, lr por defecto):** con las 80 capas (890,880 parámetros, checkpoint de 3.41 MB) da **DINOv2 = 0.5227** y **drift = 0.0732**, un claro *underfit* frente a LoRA $r=32$ (**0.6815** / 0.2721). Añadiendo a `target_modules` el `to_out` de los 20 bloques *single-stream* (+61,440 parámetros, un **6.9%** más) sube a **0.6015** (**+15%**) con drift 0.1082 en el mismo tiempo. En el benchmark oficial (L40S): **0.5763** / drift 0.1069.

**Limitaciones:**
* **Qué no prueban:** solo modelos de lenguaje (BERT, RoBERTa y OPT, de 110M a 6.7B) en tareas de texto, no modelos de difusión; su *low-data* son ~1,000 muestras, lejos de un *few-shot* de imagen. El LoRA de OPT-6.7B queda a nivel de azar (50.5% en SST-2), lo que hace dudar de ese baseline. Dejan para trabajo futuro cómo se relaciona el valor del score con el rendimiento.
* **Supuestos:** un bias suma el mismo vector a la salida para cualquier entrada ($y = Wx + b$): desplaza las activaciones pero no puede cambiar cómo la capa transforma la entrada. De ahí el *underfit* en `FLUX.2-klein` (reportado por Bossan), no solo de los pocos parámetros. El paper indica que el lr decide cuánto diverge el bias ($b_v$ rinde mejor con `1e-3` que con `1e-4` en CB); en mis runs usé el lr por defecto (`1e-4`), así que esa palanca quedó sin probar.
* **Coste:** el warmup es un entrenamiento extra (aunque el hallazgo de $b_v$ permite saltárselo). En `FLUX.2-klein`, aunque el adaptador pesa solo 3.41–3.65 MB, la memoria (~9.45 GB) y el tiempo (~990 s) son parecidos a los de LoRA (10.37 GB / 1,161 s), porque los 4B de parámetros base ocupan la mayoría de la VRAM.

**Conexión:** `#peft` `#bias-tuning` `#bitfit` `#low-data` `#flux2-klein` `#underfitting` `#tweft` (referencia de mínima capacidad expresiva en el benchmark; su warmup + selección de la mejor familia es la base de TWEFT, que lo aplica a matrices de pesos completas en vez de a biases).

**Preguntas**
1. **Conceptual:** *¿Por qué el criterio de BEFT elige mejor qué bias entrenar que la magnitud del cambio o la información de Fisher?*
   * **Respuesta:** Porque la magnitud ($\ell_1$) solo mide cuánto cambió el bias y da el mismo score a biases distintos sobre un mismo rombo aunque hayan rotado, y la Fisher empírica se calcula antes de entrenar y da un ranking estático, igual de *low-data* a *all-data*. BEFT, tras el warmup, mide longitud y ángulo a la vez con la proyección, y su ranking se adapta a la tarea y a la cantidad de datos.
2. **Matemática / Diseño:** *¿Por qué el denominador del score es $`\max(\|b^{\text{pre}}\|_2^2, \|b^{\text{post}}\|_2^2)`$ y no siempre $`\|b^{\text{pre}}\|_2^2`$?*
   * **Respuesta:** Porque así siempre se proyecta el vector más corto sobre el más largo y el ratio ($`\frac{\|b_{\text{corto}}\|_2}{\|b_{\text{largo}}\|_2}\cos\alpha`$) queda acotado por 1. Si se proyectara siempre sobre el original, un bias que triplica su norma casi sin rotar daría un ratio cercano a 3 y un score negativo, como si no hubiera cambiado. Con el máximo, tanto encoger como crecer o rotar suben el score $1 - \text{ratio}$.
3. **Crítica / Trade-offs:** *¿Por qué BEFT hace underfit en `FLUX.2-klein`, por qué su drift bajo no es una ventaja y por qué apenas ahorra VRAM frente a LoRA?*
   * **Respuesta:** Un bias es una traslación constante que no cambia cómo la capa transforma la entrada, y aprender un sujeto visual en *few-shot* necesita más que eso. El drift bajo (0.0732) solo refleja que el modelo apenas cambió: no es una ventaja si la calidad no es la misma (DINOv2 0.52 frente a 0.68 de LoRA). La VRAM apenas baja porque la ocupan sobre todo los 4B de parámetros base congelados. El margen que queda es adaptar más capas: +6.9% de parámetros dio +15% de DINOv2.

**Flashcards**
1. **P:** ¿Qué diferencia a BEFT de BitFit?
   **R:** BitFit entrena todos los biases (0.09% en BERT-base); BEFT hace un warmup, puntúa cada familia ($b_q$, $b_k$, $b_v$) por proyección y entrena solo la ganadora en todas las capas (0.01%), normalmente $b_v$.
2. **P:** ¿Qué mide el score de BEFT?
   **R:** El cambio de longitud y de ángulo de cada bias tras el warmup, proyectando el vector más corto sobre el más largo: $`1 - \frac{b^{\text{pre}} \cdot b^{\text{post}}}{\max(\|b^{\text{pre}}\|_2^2, \|b^{\text{post}}\|_2^2)}`$, promediado en todas las capas.
3. **P:** ¿Cómo aplica `peft` BEFT a `FLUX.2-klein`, cuyas capas no tienen bias?
   **R:** No hace la selección de $b_v$: añade un bias nuevo inicializado a 0 en cada capa de `target_modules`, así que entrena tantos parámetros como la suma de sus `out_features` (890,880 en 80 capas).
4. **P:** ¿Qué consiguió ampliar `target_modules` en BEFT sobre `FLUX.2-klein`?
   **R:** Añadir el `to_out` de los 20 bloques *single-stream* (+6.9% de parámetros) subió el DINOv2 de 0.5227 a 0.6015 (+15%), aunque sigue por debajo de LoRA $r=32$ (0.6815).
