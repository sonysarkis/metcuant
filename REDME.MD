## 📘 Descripción del Funcionamiento del Programa

Este programa en Python, diseñado con **Programación Orientada a Objetos (POO)**, resuelve problemas de **Programación Lineal** mediante una simulación del **método de planos de corte**.

### 🧩 Modelo LP (`LinearProgrammingModel`)
Permite al usuario definir interactivamente:
- Variables de decisión
- Función objetivo: *maximizar* o *minimizar*
- Restricciones del problema

Internamente, el modelo utiliza la librería `pulp` para construir y resolver el sistema.

### ✂️ Método de Planos de Corte Simplificado (`solve_model_with_cutting_planes()`)

El método sigue una lógica iterativa:

- **Iterativo**: Resuelve el modelo como una programación lineal continua en cada paso.
- **Verificación**: Comprueba si todas las variables tienen valores enteros.
- **Cortes heurísticos**:
  - Si alguna variable no es entera (ej. 2.5), se agrega una restricción tipo `x <= 2`.
  - Estos cortes son simplificaciones, no cortes Gomory formales.
- **Convergencia**:
  - El proceso se repite hasta que se encuentra una solución entera o se determina que no existe ninguna viable.

---

Puedes seguir extendiendo el README con secciones como `Instalación`, `Ejemplos`, o `Licencia` si lo vas a publicar. ¿Quieres ayuda para completar esas secciones también?
