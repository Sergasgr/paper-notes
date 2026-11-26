# Learning to Reason in 13 Parameters — TinyLoRA (2026, FAIR at Meta / Cornell / CMU — Morris et al.)

**Pregunta con la que lo leí:** ¿Cómo baja TinyLoRA el ajuste de todo un modelo hasta un solo parámetro, por qué solo funciona con RL, y qué pasa al entrenarlo con la loss supervisada de difusión en `FLUX.2-klein`?

**Problema:** LoRA redujo drásticamente los parámetros a reentrenar, pero incluso con $r=1$ sigue ajustando millones de parámetros (al menos 3M en Llama3-8B), porque el tamaño de $A$ y $B$ depende de la anchura del modelo; LoRA-XS rompe esa dependencia, pero sigue necesitando al menos un parámetro por módulo (cientos en un modelo grande).

**Idea clave:** Sustituir las matrices de LoRA por un vector minúsculo $v$ que modula, a través de matrices aleatorias fijas, las direcciones SVD congeladas de cada capa, y compartir ese mismo $v$ entre módulos: con $u=1$ y todos los módulos compartiéndolo, el ajuste de todo el modelo se reduce a **un único parámetro**. Solo funciona tan bien con RL.

**Método:**
TinyLoRA parte de LoRA-XS ($W' = W + U\Sigma R V^\top$, que solo entrena una pequeña matriz $R$ de $r \times r$, una "mesa de mezclas" que recombina lo que el modelo ya sabe) y sustituye $R$ por una combinación de matrices aleatorias fijas:

```math
W' = W + U\Sigma \left( \sum_{i=1}^{u} v_i P_i \right) V^\top, \qquad P_i \in \mathbb{R}^{r \times r} \text{ fijas}
```

* **Andamio congelado (SVD):** descompone la capa original en sus direcciones principales ($U$, $\Sigma$, $V$), truncadas a rango $r$, y las congela por completo para preservar el conocimiento preentrenado.
* **Vector $v$ y proyección $P$:** sustituye las matrices de LoRA por un pequeño vector entrenable $v$ de dimensión $u$, que se proyecta a través de matrices aleatorias fijas $P_i$ para que encaje con las dimensiones de la SVD. $u$ son los "interruptores" disponibles: a mayor $u$, más flexibilidad tiene el módulo para alterar su comportamiento original.
* **Compartición de pesos (*weight tying*):** obliga a varios módulos a usar exactamente el mismo $v$ ($n_{\text{tie}}$ módulos por vector, $n m u / n_{\text{tie}}$ parámetros en total). Si todas las capas lo comparten y $u=1$, se entrena 1 único parámetro. Para un presupuesto fijo, el paper recomienda reducir $u$ (hasta 1) antes que compartir $v$ entre más módulos: rinden mejor pocos parámetros únicos que muchos compartidos.
* **Por qué RL:** el SFT obliga al modelo a absorber y memorizar mucha información (formatos, estilos) junto con la solución, sin poder distinguir qué características son relevantes. Con RL, el modelo explora y solo recibe la recompensa, una señal mucho más escasa pero limpia: lo que se correlaciona con la recompensa se refuerza y el resto (tokens como "Entonces", que a veces acompañan a un acierto y a veces a un fallo) se promedia como ruido. Eso permite aprender con una capacidad minúscula.

**Resultado clave:**
* **GSM8K (Qwen2.5-7B-Instruct + GRPO):** **91%** entrenando solo **13 parámetros** (26 bytes en bf16), a menos de un 5% de *full fine-tuning*; con 10,000 parámetros pasa de 76% a 95%.
* **Matemáticas más difíciles (MATH500, AIME, AMC…):** con 196 parámetros retiene el **87%** de la mejora absoluta (media de seis benchmarks); en general recupera el 90% de la mejora entrenando 1000x menos parámetros.
* **SFT frente a RL:** con SFT hacen falta actualizaciones entre **100 y 1000 veces mayores** para el mismo rendimiento; con menos de 100 parámetros, el SFT apenas supera al modelo base.
* **Mis experimentos (`FLUX.2-klein`, loss supervisada de difusión, sin RL):** la entrada ($r=16$, $u=256$, 20,480 parámetros, 1,872 veces menos que LoRA) da **DINOv2 = 0.4354** / drift 0.0933 frente a LoRA $r=32$ (0.6815 / 0.2721). Con 13 parámetros ($u=13$, *tying* total) el drift es casi 0 (0.0086) y su ~0.37 es prácticamente lo que da el modelo base. Subiendo $r$ y $u$ (mismo lr `1e-4`): $r=64, u=256$ → 0.5403 / 0.1271; $r=128, u=512$ → **0.6043** / 0.2325 (40,960 parámetros).

**Limitaciones:**
* **Qué no prueban:** solo razonamiento matemático con RL; no saben si se generaliza a ciencia o escritura creativa. Además Qwen necesita unas 10 veces menos parámetros que LLaMA para el mismo rendimiento (con menos de 5 parámetros, LLaMA apenas mejora). Nada de difusión.
* **Supuestos:** que se entrena con RL. En este benchmark se entrena con la loss supervisada de difusión, justo el caso en el que el paper dice que no funciona: como dijo Bossan, está claro que el modelo no aprende realmente nada, pero no es sorprendente para este método. En mis pruebas sí importa $r$: con el mismo $u=64$, $r=8$ mejora a $r=2$ (0.4243 frente a 0.3771), lo contrario que en el paper (con RL, un $r$ mayor que 2 empeora y usan $r=2$). Con lr `1e-2` el modelo se rompe (DINOv2 0.05, drift 0.4958).
* **Coste:** el checkpoint pesa mucho más que los parámetros entrenables porque guarda los factores SVD congelados ($r \times (\text{in} + \text{out})$ por módulo) y las $P$ ($u \times r \times r$ por capa, con `save_projection: true`): 9.3 MiB con 5,120 parámetros y **3.1 GiB** con $r=128, u=512$ (las $P$ son 2.5 de esos 3.1 GiB, unas 21 veces el checkpoint de LoRA), con más memoria que LoRA (13,192 MB). Bossan cerró mi PR que subía la entrada a $r=64, u=256$: TinyLoRA está pensado para entrenar muy pocos parámetros, así que no tiene sentido subir más $r$, y la entrada ($r=16$, $u=256$) se mantiene.

**Conexión:** `#peft` `#rl` `#grpo` `#extreme-peft` `#svd` `#weight-sharing` `#lora-xs` `#flux2-klein` (evolución de LoRA-XS; comparte parámetros como Uni-LoRA, pero aquí $r$ es el tamaño del subespacio SVD congelado en el que actúa $v$ y subirlo no añade compartición, por eso en FLUX subir $r$ ayudó y en Uni-LoRA empeoró; con $r=128, u=512$ iguala a BEFT ampliado (0.6015) con más del doble de drift).

**Preguntas**
1. **Conceptual:** *¿Por qué TinyLoRA funciona con RL y no con SFT en régimen de muy pocos parámetros?*
   * **Respuesta:** Porque minimizar la loss de SFT obliga al modelo a absorber muchos bits (formato, estilo, cada token del ejemplo es igual de "informativo"), de los que solo una fracción es relevante para la tarea, y unos pocos parámetros no tienen capacidad para eso. En RL la información útil se reduce a la recompensa: lo que acompaña por igual a aciertos y fallos se cancela, y solo queda la señal correlacionada con el éxito, que cabe en una actualización minúscula.
2. **Matemática / Diseño:** *¿Cómo se pasa de LoRA a LoRA-XS y de ahí a TinyLoRA para llegar a un solo parámetro?*
   * **Respuesta:** LoRA entrena $A$ y $B$, cuyo tamaño depende de la anchura del modelo. LoRA-XS congela $U\Sigma V^\top$ de la SVD y solo entrena $R$ ($r \times r$): con $r=1$ queda un parámetro por módulo (cientos en total). TinyLoRA sustituye $R$ por $\sum_i v_i P_i$ con $P_i$ aleatorias fijas (solo $u$ parámetros por módulo) y comparte $v$ entre módulos; con $u=1$ y *tying* total, 1 parámetro.
3. **Crítica / Trade-offs:** *En `FLUX.2-klein`, subir $r$ y $u$ mejora el DINOv2 pero también el drift. ¿Cómo se comparan esas runs y cuál es el mejor punto?*
   * **Respuesta:** El drift solo se puede comparar entre runs con un DINOv2 parecido, porque un modelo que aprende más también cambia más; lo que cuenta es cuánto drift cuesta cada punto de DINOv2. El mejor punto es pasar de la entrada a $r=64, u=256$: +0.105 de DINOv2 por +0.034 de drift con los mismos 20,480 parámetros. Subir $u$ a 512 da +0.034 por +0.069, y $r=128$ +0.030 por +0.036 (casi igual en validación). Aun así, el límite principal es entrenar sin RL, y el checkpoint crece con $u \cdot r^2$.

**Flashcards**
1. **P:** ¿Cuál es la ecuación de actualización de TinyLoRA?
   **R:** $W' = W + U\Sigma\left(\sum_{i=1}^{u} v_i P_i\right)V^\top$: $U, \Sigma, V$ de la SVD truncada (congelados), $P_i$ aleatorias fijas y solo $v \in \mathbb{R}^u$ entrenable.
2. **P:** Para un presupuesto fijo de parámetros, ¿qué recomienda el paper: reducir $u$ o compartir $v$ entre más módulos?
   **R:** Reducir $u$ (hasta 1) antes que aumentar $n_{\text{tie}}$: rinden mejor pocos parámetros únicos que muchos compartidos.
3. **P:** ¿Por qué el checkpoint de TinyLoRA pesa 9.3 MiB con solo 5,120 parámetros?
   **R:** Porque guarda los factores SVD congelados ($r \times (\text{in} + \text{out})$ por módulo) y las matrices $P$ (`save_projection: true`); por eso crece con $r$ (y las $P$ con $u \cdot r^2$).
4. **P:** ¿Qué pasó con TinyLoRA de 13 parámetros en `FLUX.2-klein`?
   **R:** Drift casi 0 (0.0086) y DINOv2 ~0.37, lo mismo que el modelo base: sin RL, con la loss supervisada de difusión, el modelo apenas cambia.
