
# Guía de Contribución y Estilo de Commits

Para asegurar que el historial de versiones sea claro, legible y consistente, todo el equipo debe adherirse a las siguientes reglas para los mensajes de commit.

## Formato del Mensaje de Commit

Cada mensaje de commit debe seguir la siguiente estructura:

```
<tipo>(<ámbito>): <asunto>
```

**Ejemplo:** `feat(str): agregar función de pérdida ctc`

--- 

### **1. Tipo (`<tipo>`)**

El tipo debe ser uno de los siguientes. Esto indica la naturaleza del cambio.

-   **`feat`**: Para una nueva característica (feature) que se añade al proyecto.
-   **`fix`**: Para la corrección de un error (bug).
-   **`docs`**: Para cambios exclusivos en la documentación (README, guías, etc.).
-   **`style`**: Para cambios que no afectan el significado del código (espacios, formato, puntos y comas faltantes, etc.).
-   **`refactor`**: Para un cambio en el código que no corrige un error ni añade una característica (ej. renombrar una variable, simplificar una función).
-   **`perf`**: Para un cambio que mejora el rendimiento.
-   **`test`**: Para añadir o corregir pruebas existentes.
-   **`build`**: Para cambios que afectan al sistema de construcción o a dependencias externas (ej. `requirements.txt`, configuración de entorno).
-   **`chore`**: Para otros cambios que no modifican el código fuente o las pruebas (ej. actualizar el `.gitignore`).

### **2. Ámbito (`<ámbito>`)**

El ámbito es opcional y proporciona contexto sobre qué parte del proyecto se está modificando. Para este proyecto, usen los siguientes ámbitos:

-   **`str`**: Cambios relacionados con el modelo de Reconocimiento de Texto.
-   **`sentiment`**: Cambios relacionados con el modelo de Análisis de Sentimiento.
-   **`pipeline`**: Cambios en el script de integración (`main.py`) o el flujo de datos.
-   **`data`**: Cambios en los scripts de carga, preprocesamiento o generadores de datos.
-   **`setup`**: Cambios en la configuración del repositorio, entorno o dependencias.
-   **`docs`**: Para cambios en la documentación.

### **3. Asunto (`<asunto>`)**

El asunto es una descripción corta y concisa del cambio.

-   Usar el modo imperativo (ej. "agregar", "corregir", "cambiar" en lugar de "agregado", "corregido", "cambios").
-   Empezar con minúscula.
-   No terminar con un punto.

--- 

### **Ejemplos de Buenos Mensajes de Commit**

-   `feat(str): implementar arquitectura base cnn-rnn`
-   `fix(data): corregir error de path en el generador de imágenes`
-   `docs(readme): actualizar instrucciones de ejecución`
-   `refactor(sentiment): simplificar función de limpieza de texto`
-   `build: agregar jiwer y nltk a requirements.txt`
-   `test(str): añadir pruebas de evaluación para la métrica wer`

### **La Regla de Oro**

**Un commit, un cambio.** Hagan commits pequeños y lógicos. Un commit debe representar una única unidad de trabajo. Esto hace que sea más fácil de revisar, entender y, si es necesario, revertir.

---

## Flujo de Trabajo con Git (Branching Workflow)

Para mantener el proyecto ordenado y evitar conflictos, seguiremos un modelo de ramas basado en `main`, `develop` y ramas de características (`feature branches`).

### **Ramas Principales**

1.  **`main`**: Esta rama es la versión estable y de producción del proyecto. **Nadie debe hacer push directamente a `main`**. Solo se actualiza fusionando la rama `develop` cuando se alcanza un hito importante y estable.

2.  **`develop`**: Esta es la rama principal de desarrollo. Contiene los últimos cambios y características que se han completado. Todo el trabajo nuevo parte de esta rama.

### **Proceso de Desarrollo**

1.  **Sincronizar `develop`:**
    Antes de empezar a trabajar en una nueva tarea, asegúrate de tener la última versión de la rama `develop`.
    ```bash
    git checkout develop
    git pull origin develop
    ```

2.  **Crear una Rama de Característica (Feature Branch):**
    Crea una nueva rama a partir de `develop` para tu tarea. Nombra la rama de forma descriptiva usando el formato `tipo/descripcion-corta`.
    -   **Ejemplos:**
        -   `feat/implementar-modelo-str`
        -   `fix/corregir-bug-dataloader`
        -   `docs/actualizar-readme`

    ```bash
    git checkout -b feat/nombre-de-tu-caracteristica
    ```

3.  **Trabajar en la Tarea:**
    Realiza los cambios en tu rama. Haz commits pequeños y atómicos siguiendo las reglas de estilo mencionadas anteriormente.

    ```bash
    # Haces tus cambios...
    git add .
    git commit -m "feat(str): agregar capa convolucional inicial"
    # Repites el proceso...
    ```

4.  **Subir la Rama a GitHub:**
    Cuando hayas terminado tu trabajo (o quieras un backup), sube tu rama al repositorio remoto.
    ```bash
    git push -u origin feat/nombre-de-tu-caracteristica
    ```

5.  **Crear un Pull Request (PR):**
    -   Ve al repositorio en GitHub.
    -   Verás una notificación para crear un **Pull Request** desde tu rama hacia `develop`.
    -   **Asigna al menos a un miembro del equipo para que revise tu código.**
    -   Describe los cambios realizados en el PR.

6.  **Revisión de Código y Fusión (Merge):**
    -   El revisor asignado comprobará el código en busca de errores, mejoras y cumplimiento de las guías.
    -   Si hay comentarios, haz los cambios solicitados en tu rama y vuelve a subirla. El PR se actualizará automáticamente.
    -   Una vez que el PR es aprobado, el responsable (o tú mismo, si tienes permiso) fusionará la rama con `develop` usando la opción **"Squash and merge"** en GitHub. Esto condensa todos los commits de tu rama en uno solo en `develop`, manteniendo el historial limpio.

7.  **Limpieza:**
    Después de fusionar, puedes eliminar tu rama de característica tanto en el repositorio remoto como en tu local.
    ```bash
    # En local
    git checkout develop
    git branch -d feat/nombre-de-tu-caracteristica
    ```
