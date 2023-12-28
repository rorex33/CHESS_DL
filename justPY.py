from sympy import symbols, simplify, Rational, sqrt, cancel

# Определяем символьные переменные
z = symbols('z')

# Задаем выражение
expression = (z**4 - 4*sqrt(3)*z**3 + 18*z**2 - 12*sqrt(3)*z + 10) / (2*sqrt(3)*z**2 - 24 + 6*sqrt(3))

# Преобразуем числитель и знаменатель в форму, удобную для дальнейшей работы
numerator = z**4 - 4*sqrt(3)*z**3 + 18*z**2 - 12*sqrt(3)*z + 10
denominator = 2*sqrt(3)*z**2 - 24 + 6*sqrt(3)

# Упрощаем числитель и знаменатель
numerator = simplify(numerator)
denominator = simplify(denominator)

# Выражение после первичного упрощения
simplified_expression = numerator / denominator

# Продолжаем упрощать рациональное выражение
simplified_expression = cancel(simplified_expression)

# Вывод упрощенного выражения
print("Упрощенное выражение:")