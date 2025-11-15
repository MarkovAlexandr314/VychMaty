import numpy as np
import enum
import time
import matplotlib.pyplot as plt

plt.style.use("ggplot")

#  Сохраняем времена работы
time_dict = {}
#  Сохраняем невязки
error_dict = {}

class Method(enum.Enum):
    '''
        Перечисление для удобного выбора метода
    '''
    GAUSS       = 1
    GAUSS_TWO   = 2
    ORTHOG      = 3
    RUN         = 4
    LU          = 5
    ITER_SIMPLE = 6
    ITER_YKOBI  = 7
    ITER_ZEIDEL = 8
    ITER_MIN_R  = 9
    NP_SOLVER  = 10

class Solver:
    """
        Решатель СЛАУ
    """
    def time_and_error(func):
        """
            Декоратор для замеров времени выполнения и невязки 
        """
        def wrapped(self, *args):
            start_time = time.perf_counter_ns()
            res = func(self, *args)
            end_time = time.perf_counter_ns()
            name = func.__name__
            if name not in time_dict:
                time_dict[name] = []
            if name not in error_dict:
                error_dict[name] = []
            time_dict[name].append(end_time - start_time)
            error_dict[name].append(np.linalg.norm(args[0] @ res - args[1]))
            return res
        wrapped.__name__ = func.__name__
        return wrapped

    def solve(self, A: np.array, f: np.array, method: Method=Method.GAUSS, x0=None, n=None, P=None) -> np.array:
        '''
        Решить СЛАУ

        Ключевые аргументы:
            A       - матрица системы
            vec     - вектор правых частей
            method  - метод, применяемый для вычислений(по умолчанию Method.GAUSS)
        Аргументы для итерационных методов:
            x0      - начальное приближение решения
            n       - число итераций
            P       - некоторая несингулярная матрица, нужна для простой итерации

        Типы метод: 
            Method.GAUSS        - метод Гаусса
            Method.GAUSS_TWO    - метод Гаусса с выбором главного элемента по строкам
            Method.ORTHOG       - метод ортогонализации
            Method.RUN          - метод прогонки
            Method.LU           - метод LU-разложения
            Method.ITER_SIMPLE  - метод простой итерации
            Method.ITER_YKOBI   - метод Якоби
            Method.ITER_ZEIDEL  - метод Зейделя
            Method.ITER_MIN_R   - метод минимальных невязок
        '''
        if method == Method.GAUSS:
            answer = self.GAUSS_solve(A, f)
        elif method == Method.GAUSS_TWO:
            answer = self.GAUSS_TWO(A, f)
        elif method == Method.ORTHOG:
            answer = self.orthogonalize_method(A, f)
        elif method == Method.RUN:
            answer = self.RUN(A, f)
        elif method == Method.LU:
            answer = self.LU_solve(A, f)
        elif method == Method.ITER_SIMPLE:
            answer = self.iter_simple(A, f, P, x0, n)
        elif method == Method.ITER_ZEIDEL:
            answer = self.iter_ZEIDEL(A, f, x0, n)
        elif method == Method.ITER_YKOBI:
            answer = self.iter_YKOBI(A, f, x0, n)
        elif method == Method.ITER_MIN_R:
            answer = self.iter_MIN_R(A, f, x0, n)
        elif method == Method.NP_SOLVER:
            answer = self.np_solve(A, f)



        #  Допустимое отклонение невязки
        # r_delta = 1e-5
        #  Проверяем невязку
        # r = np.linalg.norm(A @ answer - f)
        # if r > r_delta:
            # raise RuntimeError(f"Discrepancy {r} > {r_delta}; Dim: {A.shape[0]}")

        return answer

    @time_and_error
    def np_solve(self, A: np.array, f: np.array) -> np.array:
        return np.linalg.solve(A, f)

    @time_and_error
    def GAUSS_solve(self, A: np.array, f: np.array) -> np.array:
        """
            Решение СЛАУ методом Гаусса

            Параметры:
                A - матрица СЛАУ
                f - вектор правых частей
    
            Возвращает:
                x - решение СЛАУ
        """
        #  Расширенная матрица
        extended_A = np.append(A, f, axis=1)
        #  Прямой ход метода Гаусса
        dim = extended_A.shape[0]
        for j in np.arange(dim-1):
            extended_A[j, j:] /= extended_A[j, j]
            
            # коэфф-ты на которые умножится j-я строка
            j_column = extended_A[j+1:, j]
            j_column = j_column[..., None]
            extended_A[j+1:, j:] -= extended_A[j, j:] * j_column
            
        #  Обратный ход метода Гаусса
        extended_A[dim-1, dim-1:] /= extended_A[dim-1, dim-1]
        for j in np.arange(dim-1, 0, -1):
            j_column = extended_A[:j, j]
            j_column = j_column[..., None]
            extended_A[:j, j:] -= extended_A[j, j:] * j_column
        
        return extended_A[:, -1][..., None]

    def __LU(self,A):
        """
            LU-разложение матрицы
                        
            Параметры:
                A - матрица СЛАУ
    
            Возвращает:
                L - нижняя треугольная матрица
                U - верхняя треугольная матрица
        """
        n = A.shape[0]
        L = np.eye(n).astype(float)
        U = np.zeros_like(A).astype(float)
        for i in range(n):
            # Сначала вычисляем строку U
            for j in range(i, n):
                U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])

            # Затем вычисляем столбец L
            for j in range(i+1, n):
                L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]

        return L, U

    @time_and_error
    def LU_solve(self, A, f):
        """
            Решение СЛАУ методом LU-разложения
                        
            Параметры:
                A - матрица СЛАУ
                f - вектор правых частей
    
            Возвращает:
                x - решение СЛАУ
        """
        # Выполняем LU-разложение матрицы A
        L, U = self.__LU(A)
        n = A.shape[0]

        # Прямая подстановка: Ly = f
        y = np.zeros(n)
        for i in np.arange(n):
            y[i] = f[i] - L[i, :i] @ y[:i]

        # Обратная подстановка: Ux = y
        x = np.zeros(n)
        for i in np.arange(n-1, -1, -1):
            x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]    
        
        return x[..., None]

    @time_and_error
    def GAUSS_TWO(self, A, f):
        """
            Решение СЛАУ методом Гаусса с выбором главного элемента по строкам
                        
            Параметры:
                A - матрица СЛАУ
                f - вектор правых частей
    
            Возвращает:
                x - решение СЛАУ
        """
        #  Расширенная матрица
        extended_A = np.append(A, f, axis=1).astype(float)

        dim = extended_A.shape[0]
        for i in np.arange(dim-1):
            #  Ищем индекс самого большого модулю элемента в текущем столбце
            ind_max = i + np.argmax(np.abs(extended_A[i:, i]))

            #  Меняем местами текущую строку со строкой ind_max
            extended_A[[i, ind_max]] = extended_A[[ind_max, i]]

            #  Вычисляем поправочные множители
            i_column = extended_A[i+1:, i] / extended_A[i, i]
            i_column = i_column[..., None]
            extended_A[i+1:, i:] -= extended_A[i, i:] * i_column

        #  Обратный ход метода Гаусса
        for j in np.arange(dim-1, -1, -1):
            extended_A[j, :] /= extended_A[j, j]
            j_column = extended_A[:j, j]
            j_column = j_column[..., None]
            extended_A[:j, j:] -= extended_A[j, j:] * j_column

        return extended_A[:, -1][..., None]

    def __get_c(self, A):
        """ Получаем наддиагональные элементы """
        a_mask = np.zeros_like(A)
        a_mask[:-1, 1:] = np.eye(A.shape[0]-1)
        return A[np.where(a_mask != 0)]

    def __get_b(self, A):
        """ Получаем диагональные элементы """
        tmp = A * np.eye(A.shape[0])
        b = tmp[np.where(tmp != 0)]
        return b

    def __get_a(self, A):
        """ Получаем поддиагональные элементы """
        c_mask = np.zeros_like(A)
        c_mask[1:, :-1] = np.eye(A.shape[0]-1)
        return A[np.where(c_mask != 0)]

    @time_and_error
    def RUN(self, A, f):
        """
            Решение СЛАУ методом прогонки
            
            Параметры:
                A - матрица СЛАУ
                f - вектор правых частей
    
            Возвращает:
                x - решение СЛАУ
        """
        n = A.shape[0]

        #  Получим коэфф-ты a, b, c
        a = self.__get_a(A)
        b = self.__get_b(A)
        c = self.__get_c(A)

        #  Вектор ответов
        x = np.zeros(n)
        #  Вспомогательные векторы
        B_ = np.zeros(n)
        F_ = np.zeros(n)

        #  Считаем вспомогательные векторы
        B_[0] = b[0]
        for i in range(1, n):
            B_[i] = b[i] - a[i-1] / B_[i-1] * c[i-1]

        F_[0] = f[0]
        for i in range(1, n):
            F_[i] = f[i] - a[i-1] / B_[i-1] * F_[i-1]

        #  Считаем ответ
        x[n-1] = F_[n-1] / B_[n-1]
        for i in range(n-2, -1, -1):
            x[i] = (F_[i] - c[i]*x[i+1]) / B_[i]

        return x

    def __orthogonalize_rows(self, A, f):
        """
            Ортогонализация матрицы СЛАУ
        """
        n = A.shape[0]

        # Создаем расширенную матрицу
        Af = np.column_stack((A, f)).astype(np.float64)

        # Процесс ортогонализации строк (метод Грама-Шмидта)
        for i in range(n):
            # Нормируем текущую строку
            norm = np.linalg.norm(Af[i, :n])

            Af[i] = Af[i] / norm

            # Ортогонализуем последующие строки относительно текущей
            for j in range(i + 1, n):
                projection = np.dot(Af[j, :n], Af[i, :n])
                Af[j] = Af[j] - projection * Af[i]
        return Af

    @time_and_error
    def orthogonalize_method(self, A, f):
        """
            Решение СЛАУ методом ортогонолизации

            Параметры:
                A - матрица СЛАУ
                b - вектор правых частей
    
            Возвращает:
                A_ort.T @ b_ort - решение СЛАУ
        """
        A_ort = self.__orthogonalize_rows(A, f)[:, :-1]
        f_ort = self.__orthogonalize_rows(A, f)[:, -1:]

        return A_ort.T @ f_ort
    
    @time_and_error
    def iter_simple(self, A, f, P, x0, n):
        """
            Решение СЛАУ методом простой итерации
            
            Параметры:
                A   - матрица СЛАУ
                f   - вектор правых частей
                P   - некоторая несингулярная матрица
                x0  - начальное приблиение решения
                n   - число итераций

            Возвращает:
                x - решение СЛАУ
        """
        B = (P @ A + np.eye(A.shape[0]))
        C = P @ f
        x_next = x0
        for _ in np.arange(n):
            x_prev = x_next
            x_next = B @ x_prev + C

        return x_next
    
    @time_and_error
    def iter_ZEIDEL(self, A, f, x0, n):
        """
            Решение СЛАУ методом Зейделя
            
            Параметры:
                A   - матрица СЛАУ
                f   - вектор правых частей
                x0  - начальное приблиение решения
                n   - число итераций

            Возвращает:
                x - решение СЛАУ
        """
        # Диагональная матрица
        L = np.tril(A, k=-1)  # Нижняя треугольная без  диагонали
        U = np.triu(A, k=1)   # Верхняя треугольная без диагонали
        D = np.diag(np.diag(A))
        x_next = x0
        LD_reverse = np.linalg.inv(L + D)
        for _ in np.arange(n):
            x_prev = x_next
            x_next = LD_reverse @ (f - U @ x_prev)

        return x_next

    @time_and_error
    def iter_YKOBI(self, A, f, x0, n):
        """
            Решение СЛАУ методом Якоби
            
            Параметры:
                A   - матрица СЛАУ
                f   - вектор правых частей
                x0  - начальное приблиение решения
                n   - число итераций

            Возвращает:
                x - решение СЛАУ
        """
        # Диагональная матрица
        L = np.tril(A, k=-1)  # Нижняя треугольная без диагонали
        U = np.triu(A, k=1)   # Верхняя треугольная без диагонали
        D = np.diag(np.diag(A))
        x_next = x0
        D_reverse = np.linalg.inv(D)
        for _ in np.arange(n):
            x_prev = x_next
            x_next = D_reverse @ (f - (L + U) @ x_prev)

        return x_next

    @time_and_error
    def iter_MIN_R(self, A, f, x0, n):
        """
            Решение СЛАУ методом минимальных невязок
            
            Параметры:
                A   - матрица СЛАУ
                f   - вектор правых частей
                x0  - начальное приблиение решения
                n   - число итераций

            Возвращает:
                x - решение СЛАУ
        """
        x_next = x0
        r_next = A @ x_next - f
        #  Проверка, чтоб не делить на 0
        tmp = A @ r_next
        if((np.vdot(tmp, tmp)) == 0):
            return x_next
        tau_next = (np.vdot(r_next, tmp)) / (np.vdot(tmp, tmp))

        for _ in np.arange(n):
            x_prev = x_next
            r_prev = r_next
            tau_prev = tau_next
            x_next = x_prev - tau_prev * r_prev
            r_next = A @ x_next - f
            tmp = A @ r_next
            tau_next = (np.vdot(r_next, tmp)) / (np.vdot(tmp, tmp))
            #  Проверка, чтоб не делить на 0
            if(np.vdot(tmp, tmp) == 0):
                break

        return x_next