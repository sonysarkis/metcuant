import pulp
import numpy as np

class LinearProgrammingModel:
    def __init__(self):
        self.problem = None
        self.variables = {}
        self.objective_expression = None
        self.constraints = []

    def add_decision_variables(self):
        """
        Permite al usuario agregar variables de decisión al modelo.
        Las variables son asumidas como no negativas.
        """
        while True:
            var_name = input("Ingrese el nombre de la variable de decisión (ej. x1, x2) o 'fin' para terminar: ").strip()
            if var_name.lower() == 'fin':
                break
            if var_name in self.variables:
                print(f"La variable '{var_name}' ya existe. Elija otro nombre.")
                continue
            self.variables[var_name] = pulp.LpVariable(var_name, lowBound=0)
            print(f"Variable '{var_name}' agregada.")

    def add_objective_function(self):
        """
        Permite al usuario definir la función objetivo.
        Soporta maximización o minimización.
        Ejemplo: 5*x1 + 4*x2
        """
        if not self.variables:
            print("Primero debe agregar variables de decisión.")
            return

        obj_type_input = input("¿Es un problema de maximización (max) o minimización (min)? ").strip().lower()
        if obj_type_input == 'max':
            self.problem = pulp.LpProblem("Linear_Program", pulp.LpMaximize)
        elif obj_type_input == 'min':
            self.problem = pulp.LpProblem("Linear_Program", pulp.LpMinimize)
        else:
            print("Tipo de objetivo inválido. Debe ser 'max' o 'min'.")
            return

        while True:
            obj_str = input("Ingrese la función objetivo (ej. 5*x1 + 4*x2): ").strip()
            try:
                local_vars = {name: var for name, var in self.variables.items()}
                self.objective_expression = eval(obj_str, {"__builtins__": None}, local_vars)
                self.problem += self.objective_expression, "Objective Function"
                print("Función objetivo agregada.")
                break
            except Exception as e:
                print(f"Error al parsear la función objetivo. Asegúrese de que las variables existen y el formato es correcto. Error: {e}")

    def add_constraints(self):
        """
        Permite al usuario agregar restricciones al modelo.
        Ejemplo: 6*x1 + 4*x2 <= 24
        """
        if not self.problem:
            print("Primero debe definir la función objetivo.")
            return

        while True:
            constraint_str = input("Ingrese la restricción (ej. 6*x1 + 4*x2 <= 24) o 'fin' para terminar: ").strip()
            if constraint_str.lower() == 'fin':
                break
            
            try:
                local_vars = {name: var for name, var in self.variables.items()}
                if '<=' in constraint_str:
                    parts = constraint_str.split('<=')
                    lhs = eval(parts[0].strip(), {"__builtins__": None}, local_vars)
                    rhs = float(parts[1].strip())
                    constraint = lhs <= rhs
                elif '>=' in constraint_str:
                    parts = constraint_str.split('>=')
                    lhs = eval(parts[0].strip(), {"__builtins__": None}, local_vars)
                    rhs = float(parts[1].strip())
                    constraint = lhs >= rhs
                elif '==' in constraint_str:
                    parts = constraint_str.split('==')
                    lhs = eval(parts[0].strip(), {"__builtins__": None}, local_vars)
                    rhs = float(parts[1].strip())
                    constraint = lhs == rhs
                else:
                    raise ValueError("Operador de restricción no reconocido. Use <=, >= o ==.")
                self.problem += constraint, f"Constraint_{len(self.constraints) + 1}"
                self.constraints.append(constraint)
                print(f"Restricción '{constraint_str}' agregada.")
            except Exception as e:
                print(f"Error al parsear la restricción. Asegúrese de que las variables existen y el formato es correcto. Error: {e}")

    def solve_model_with_cutting_planes(self):
        """
        Resuelve el modelo de programación lineal utilizando un enfoque de planos de corte.
        Este es un enfoque simplificado para demostrar la idea de los planos de corte,
        particularmente para problemas de programación lineal entera (PIL).
        
        Nota: La implementación de un generador de cortes de Gomory real requiere
        acceso a la tabla simplex o información de dualidad, que pulp no expone directamente.
        Aquí simularemos la adición de "cortes" si las soluciones no son enteras,
        restringiendo los valores para forzar la integralidad.
        """
        if not self.problem:
            print("El modelo no está completamente definido. Asegúrese de agregar variables, función objetivo y restricciones.")
            return

        iteration = 0
        while True:
            iteration += 1
            print(f"\n--- Iteración de Plano de Corte {iteration} ---")
            
            status = self.problem.solve(pulp.PULP_CBC_CMD(msg=0))
            print(f"Estado de la solución LP relajada: {pulp.LpStatus[status]}")

            if status != pulp.LpStatus.Optimal:
                print("No se encontró una solución óptima para el problema relajado. Terminando.")
                break

            print("Valores de las variables en la solución LP relajada:")
            current_solution_is_integer = True
            for var_name, var_obj in self.variables.items():
                var_value = var_obj.varValue
                print(f"{var_name} = {var_value:.4f}")
                if var_value is not None and not np.isclose(var_value, round(var_value)):
                    current_solution_is_integer = False

            if current_solution_is_integer:
                print("\n¡Solución entera encontrada!")
                print(f"Valor óptimo de la función objetivo: {pulp.value(self.problem.objective):.4f}")
                break
            else:
                print("La solución actual no es entera. Generando cortes...")
                cuts_added_in_this_iteration = False
                for var_name, var_obj in self.variables.items():
                    var_value = var_obj.varValue
                    if var_value is not None and not np.isclose(var_value, round(var_value)):
                        floor_val = np.floor(var_value)
                        ceil_val = np.ceil(var_value)
                        fractional_var = None
                        for name, var in self.variables.items():
                            if var.varValue is not None and not np.isclose(var.varValue, round(var.varValue)):
                                fractional_var = var
                                break
                        if fractional_var:
                            var_value = fractional_var.varValue
                            new_constraint = fractional_var <= np.floor(var_value)
                            self.problem += new_constraint, f"Cut_Floor_{fractional_var.name}_{iteration}"
                            print(f"Agregando el 'corte' (restricción): {fractional_var.name} <= {np.floor(var_value)}")
                            cuts_added_in_this_iteration = True
                if not cuts_added_in_this_iteration:
                    print("No se encontraron más variables fraccionarias para cortar. Terminando.")
                    break
            
        print("\n--- Solución Final del Modelo de Programación Lineal Entera ---")
        if pulp.LpStatus[self.problem.status] == "Optimal":
            print(f"Estado de la solución: {pulp.LpStatus[self.problem.status]}")
            print(f"Valor óptimo de la función objetivo: {pulp.value(self.problem.objective):.4f}")
            for var_name, var_obj in self.variables.items():
                print(f"{var_name} = {var_obj.varValue:.4f}")
        else:
            print("No se pudo encontrar una solución óptima entera.")


def main_menu():
    model = LinearProgrammingModel()
    
    while True:
        print("\n--- Menú de Programación Lineal ---")
        print("1. Agregar Variables de Decisión")
        print("2. Definir Función Objetivo")
        print("3. Agregar Restricciones")
        print("4. Resolver Modelo (Plano de Cortes Simplificado)")
        print("5. Salir")
        
        choice = input("Ingrese su opción: ").strip()

        if choice == '1':
            model.add_decision_variables()
        elif choice == '2':
            model.add_objective_function()
        elif choice == '3':
            model.add_constraints()
        elif choice == '4':
            model.solve_model_with_cutting_planes()
        elif choice == '5':
            print("Saliendo del programa.")
            break
        else:
            print("Opción inválida. Por favor, intente de nuevo.")

if __name__ == "__main__":
    main_menu()