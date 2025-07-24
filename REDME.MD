Este programa Python, diseñado con Programación Orientada a Objetos (POO), resuelve problemas de Programación Lineal utilizando una simulación del método de planos de corte.

Modelo LP (Clase LinearProgrammingModel): Permite al usuario definir interactivamente las variables de decisión, la función objetivo (maximizar/minimizar) y las restricciones del problema de programación lineal. Internamente, utiliza la librería pulp para construir el modelo.


Método de Planos de Corte Simplificado (solve_model_with_cutting_planes()):

Iterativo: Resuelve el problema como una Programación Lineal Continua en cada paso.

Verificación: Comprueba si la solución actual tiene valores enteros para todas las variables.

"Cortes" Heurísticos: Si la solución no es entera, añade una nueva restricción ("corte") que elimina el punto fraccionario actual de la región factible (por ejemplo, si una variable es 2.5, añade x <= 2). Este es un corte simplificado, no un corte de Gomory riguroso.

Convergencia: El proceso se repite hasta encontrar una solución entera o determinar que no hay una.

El programa ofrece un menú interactivo en consola para que el usuario ingrese todos los datos del modelo paso a paso y observe la resolución.