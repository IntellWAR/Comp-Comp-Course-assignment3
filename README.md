# Задача коммивояжера
Assignment 3 for the course of Computational Complexity

## Структура проекта
Каждый файл содержит отдельные части решения и запускается отдельно.

- *main.py* - содержит первую часть задания с решением путём полного перебора;
- *gen_rand_big_matrix.py* - код для генерации симметричной тестовой матрицы любого заданного размера;
- *Annealing.py* - моё аппроксимационное решение через мета-евристику;
- *integer_programming.py* - точное решение задачи, найденное на просторах хабра (https://habr.com/ru/articles/839804/), для сравнения с аппроксимационным решением в последней части задания.


#### Part 1
1. Классическая NP-трудная задача, заключающаяся в поиске самого короткого пути, проходящего через каждый из указанных городов хотя бы по одному разу. Причём путь обязан замыкаться в стартовом городе.

2. В качестве абстракции можно представить граф, где вершины - города, а рёбра - дороги между ними с указанной длиной. В нашем случае этот граф представлен в виде массива, где номер столбца или строки - это номер города, а элемент на пересечении выбранных городов - это длина дороги между ними. Т.е.
```a[i][j] = l```, где `i`, `j` - номера городов, а `l` - длина дороги между ними.
Пример массива с 4-мя городами (примем расстояние между городами симметричным):

| i\j | 0   | 1   | 2   | 3   |
| --- | --- | --- | --- | --- |
| 0   | 0   | 10  | 15  | 20  |
| 1   | 10  | 0   | 35  | 25  |
| 2   | 15  | 35  | 0   | 30  |
| 3   | 20  | 25  | 30  | 0   |

Тестовые массивы вставлены в код вручную.

3. Используется метод перебора, потому вычислительная сложность складывается из: 
- Сложности самого перебора всех возможных путей, равного количеству перестановок городов (O(n!) = Ω(n!) => Θ(n!));
- Сложности расчёта длины каждого пути (O(n) = Ω(n) => Θ(n));

Тогда общая сложность алгоритма - Θ(n! $*$ n).

Реальное затраченное время:
```
Test Case 1:
  Best Route: (0, 1, 3, 2)
  Minimum Cost: 80
  Expected Cost: 80
  Valid Solution: True
  Time Taken: 0.0000 seconds

Test Case 2:
  Best Route: (0, 2, 3, 1, 4)
  Minimum Cost: 34
  Expected Cost: 34
  Valid Solution: True
  Time Taken: 0.0000 seconds

Test Case 3:
  Best Route: (0, 2, 4, 1, 3, 5)
  Minimum Cost: 44
  Expected Cost: 44
  Valid Solution: True
  Time Taken: 0.0010 seconds

Test Case 4:
  Best Route: (0, 1, 5, 6, 4, 3, 2)
  Minimum Cost: 62
  Expected Cost: 62
  Valid Solution: True
  Time Taken: 0.0030 seconds

Test Case 5:
  Best Route: (0, 1, 7, 4, 2, 3, 5, 6)
  Minimum Cost: 115
  Expected Cost: 115
  Valid Solution: True
  Time Taken: 0.0269 seconds

Test Case 6:
  Best Route: (0, 1, 3, 5, 6, 8, 2, 4, 7)
  Minimum Cost: 145
  Expected Cost: 145
  Valid Solution: True
  Time Taken: 0.2633 seconds

Test Case 7:
  Best Route: (0, 7, 9, 6, 1, 4, 8, 3, 2, 5, 10, 11)
  Minimum Cost: 251
  Expected Cost: 251
  Valid Solution: True
  Time Taken: 460.7284 seconds
```
Рост времени в примерах соответствует аппроксимации $t(n) = t(1) * n * n!$

#### Part 2
**1.** Для более эффективного поиска пути выбран алгоритм Annealing.
**2**. Этот алгоритм выбран по следующим причинам:
- Он исследует пространство решений стохастически, избегая локальных минимумов;
- Постепенный процесс охлаждения обеспечивает приближение к оптимальному решению;
- Относительно легко реализуется.

Описание использования метода в коде:
  1) Сначала выбирается случайный маршрут;
  2) Определяется начальная температура (temperature), скорость охлаждения (cooling_rate) и критерии остановки (например, минимальная температура или предел повторения);
  3) Генерация соседей. На этом этапе исследуется пространство поиска, и предлагаются небольшие случайные изменения в текущем решении. Двухсторонняя замена предполагает выбор двух краев текущего маршрута и изменение порядка расположения городов между ними. При этом создается новый маршрут (соседний), который похож на текущий, но имеет несколько иную структуру. Например, маршрут [0, 1, 2, 3, 4] может превратиться в [0, 1, 4, 3, 2];
  4) Оценка качества соседа. Вычисляется разница в стоимости маршрутов (delta_cost = сосед - текущий). Если delta_cost\<0, то принимается сосед, если delta_cost\>0, то сосед принимается с вероятностью $P=exp[delta\_cost/temperature]$;
  5) После каждой итерации обновляется температура (temperature = temperature*cooling\_rate);
  6) Завершение алгоритма, если была достигнута предельно низкая температура или максимальное количество итераций.

**Оценка сложности алгоритма**: 
На каждую итерацию $O(n^2)$ из-за двухстороннего обмена и расчета стоимости маршрута. Тогда суммарна сложность - $O(k*n^2)$, где k - количество итераций (в нашем случае $k_{max}$ = 10000)

**3**.  Результаты тестов по производительности:
```
Test Case 1:
  Best Route: [1, 0, 2, 3]
  Minimum Cost: 80
  Expected Cost: 80
  Time Taken: 0.0110 seconds

Test Case 2:
  Best Route: [2, 0, 4, 1, 3]
  Minimum Cost: 34
  Expected Cost: 34
  Time Taken: 0.0110 seconds

Test Case 3:
  Best Route: [0, 5, 3, 1, 4, 2]
  Minimum Cost: 44
  Expected Cost: 44
  Time Taken: 0.0120 seconds

Test Case 4:
  Best Route: [6, 4, 3, 2, 0, 1, 5]
  Minimum Cost: 62
  Expected Cost: 62
  Time Taken: 0.0120 seconds

Test Case 5:
  Best Route: [2, 3, 0, 6, 5, 4, 7, 1]
  Minimum Cost: 115
  Expected Cost: 115
  Time Taken: 0.0150 seconds

Test Case 6:
  Best Route: [7, 2, 8, 6, 5, 3, 1, 0, 4]
  Minimum Cost: 145
  Expected Cost: 145
  Time Taken: 0.0140 seconds

Test Case 7: # 11 городов
  Best Route: [8, 4, 1, 6, 9, 7, 0, 11, 10, 5, 2, 3]
  Minimum Cost: 251
  Expected Cost: 251
  Time Taken: 0.0140 seconds
```

Также сравним с решением взятым с habr'а https://habr.com/ru/articles/839804/.

```
Time Taken: 0.0407 seconds

Min path = [(0, 11), (11, 10), (10, 5), (5, 2), (2, 3), (3, 8), (8, 4), (4, 1), (1, 6), (6, 9), (9, 7), (7, 0)]
Length = 251
```
Справедливости ради, автор в своей статье указывал, что симметричные задачи его алгоритм решает дольше, чем нессиметричные. А также на матрицах очень большого размера (N~100) применённая аппроксимация через мета-евристику даёт всё же заметную погрешность, а метод автора даёт точное решение.

Результат Annealing для N=100 (initial_temperature=1000, cooling_rate=0.9999, max_iterations=100000):
```
Best Route: [83, 19, 23, 2, 49, 27, 79, 38, 51, 66, 34, 65, 5, 93, 88, 36, 56, 52, 78, 20, 77, 16, 85, 21, 9, 39, 53, 41, 89, 60, 3, 67, 86, 92, 18, 71, 74, 30, 42, 31, 76, 99, 40, 64, 26, 87, 54, 98, 24, 94, 58, 37, 46, 32, 6, 17, 43, 90, 48, 33, 70, 62, 63, 80, 55, 72, 95, 29, 75, 81, 82, 69, 91, 7, 50, 44, 12, 11, 35, 84, 10, 97, 73, 0, 47, 96, 14, 13, 68, 45, 22, 15, 61, 1, 8, 28, 25, 59, 4, 57]
  Minimum Cost: 1207
  Expected Cost: None
  Time Taken: 1.0908 seconds
```

А вот результат алгоритма с хабра для той же матрицы:
```
Time Taken: 3.6448 seconds

Min path = [(0, 41), (41, 12), (12, 43), (43, 23), (23, 2), (2, 49), (49, 32), (32, 6), (6, 17), (17, 98), (98, 82), (82, 69), (69, 60), (60, 45), (45, 22), (22, 94), (94, 92), (92, 64), (64, 40), (40, 24), (24, 5), (5, 93), (93, 63), (63, 62), (62, 51), (51, 66), (66, 34), (34, 65), (65, 83), (83, 72), (72, 27), (27, 68), (68, 59), (59, 4), (4, 57), (57, 79), (79, 38), (38, 96), (96, 89), (89, 18), (18, 36), (36, 88), (88, 39), (39, 71), (71, 74), (74, 25), (25, 61), (61, 1), (1, 8), (8, 46), (46, 37), (37, 35), (35, 11), (11, 91), (91, 7), (7, 50), (50, 44), (44, 10), (10, 84), (84, 86), (86, 67), (67, 58), (58, 48), (48, 33), (33, 70), (70, 28), (28, 14), (14, 13), (13, 53), (53, 16), (16, 85), (85, 21), (21, 55), (55, 52), (52, 56), (56, 78), (78, 20), (20, 26), (26, 87), (87, 54), (54, 76), (76, 99), (99, 19), (19, 77), (77, 95), (95, 29), (29, 73), (73, 97), (97, 80), (80, 90), (90, 15), (15, 30), (30, 42), (42, 31), (31, 9), (9, 3), (3, 81), (81, 75), (75, 47), (47, 0)]
Length = 945
```
