import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from scipy.special import erf

P_change = 0.3
SET_SIZE = 200

E_LETTER = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 1, 1, 1, 1, 0]])

O_LETTER = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 1, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 1, 1, 1, 0, 0, 0]])


def invert_value(p, value, p_change=0.3):
    return value if p > p_change else 1 - value


def transform_matrices(letter_matrix, selection_size=200, p_change=0.3):
    letter_shape = np.shape(letter_matrix)
    matrix_as_vector = letter_matrix.flatten()
    result = np.zeros((selection_size, letter_shape[0] * letter_shape[1]))

    invert_matrix_values = np.vectorize(invert_value)
    for i in range(0, selection_size, 1):
        random_vector = np.random.uniform(0, 1, letter_shape[0] * letter_shape[1])
        result[i] = invert_matrix_values(random_vector, matrix_as_vector, p_change)

    if selection_size == 1:
        return result[0].reshape(letter_matrix.shape)
    return result


def calc_condition_probs(set_changed_vectors):
    shape_of_array = np.shape(set_changed_vectors)  # 200 x 81
    shape_result_array = np.shape(set_changed_vectors[0])

    cum_sum = np.zeros(shape_result_array)
    for i in range(0, shape_of_array[0], 1):
        cum_sum += set_changed_vectors[i]

    print(f'CUMSUM: {cum_sum}')
    return np.divide(cum_sum, shape_of_array[0])


def calc_binarySD(cond_probs_array_0, cond_probs_array_1):
    size = cond_probs_array_0.size

    part_0 = np.zeros(size)
    part_1 = np.zeros(size)

    for i in range(0, size, 1):
        log_part = calc_Wlj_coef_arr(cond_probs_array_1, cond_probs_array_0)[i]
        part_0[i] = np.power(log_part, 2) * cond_probs_array_0[i] * (1 - cond_probs_array_0[i])
        part_1[i] = np.power(log_part, 2) * cond_probs_array_1[i] * (1 - cond_probs_array_1[i])

    return np.sqrt(np.sum(part_0)), np.sqrt(np.sum(part_1))


def calc_Wlj_coef_arr(cond_probs_array_l, cond_probs_array_j):
    shape = np.shape(cond_probs_array_l)
    result = np.zeros(shape)

    for i in range(0, shape[0], 1):
        result[i] = math.log(
            ((cond_probs_array_l[i] / (1 - cond_probs_array_l[i])) *
             ((1 - cond_probs_array_j[i]) / cond_probs_array_j[i]))
        )
    return result


def calc_M(cond_probs_array_0, cond_probs_array_1):
    size = cond_probs_array_0.size

    m0_part = np.zeros(size)
    m1_part = np.zeros(size)

    wlj_array = calc_Wlj_coef_arr(cond_probs_array_1,
                                  cond_probs_array_0)
    for i in range(0, size, 1):
        m0_part[i] = wlj_array[i] * cond_probs_array_0[i]

        m1_part[i] = wlj_array[i] * cond_probs_array_1[i]

    return np.sum(m0_part), np.sum(m1_part)


def calc_small_lambda(Pl, Pj, cond_probs_array_l, cond_probs_array_j):
    shape = np.shape(cond_probs_array_l)
    result_part = np.zeros(shape)

    for i in range(0, shape[0], 1):
        result_part[i] = math.log((1 - cond_probs_array_l[i]) / (1 - cond_probs_array_j[i]))

    return math.log(Pj / Pl) + np.sum(result_part)


def classify_vectors_array(vectors_array, Pl, Pj, cond_probs_array_l, cond_pobs_array_j):
    shape = np.shape(vectors_array)
    result = np.zeros(shape[0], int)

    for i in range(0, shape[0], 1):
        result[i] = classify_Bayes(vectors_array[i], Pl, Pj, cond_probs_array_l, cond_pobs_array_j)

    return result


def classify_Bayes(X_v, Pl, Pj, cond_probs_array_l, cond_probs_array_j):
    Lamda_tilda = calc_big_lambda(X_v, cond_probs_array_l, cond_probs_array_j)
    lambda_tilda = calc_small_lambda(Pl, Pj, cond_probs_array_l, cond_probs_array_j)
    return int(0) if Lamda_tilda >= lambda_tilda else int(1)


def calc_big_lambda(X_vector, cond_probs_array_l, cond_probs_array_j):
    array_wlj = calc_Wlj_coef_arr(cond_probs_array_l, cond_probs_array_j)
    return np.sum(X_vector * array_wlj)


def calculate_exp_error(classified_array):
    return float(np.sum(classified_array) / classified_array.size)


def calc_theoretical_error(p0, p1, cond_probs_array_0, cond_probs_array_1):
    sm_lambda = calc_small_lambda(p0, p1, cond_probs_array_0, cond_probs_array_1)

    m0, m1 = calc_M(cond_probs_array_0, cond_probs_array_1)
    sd0, sd1 = calc_binarySD(cond_probs_array_0, cond_probs_array_1)

    p0 = 1 - laplas_function((sm_lambda - m0) / sd0)
    p1 = laplas_function((sm_lambda - m1) / sd1)

    return np.array([p0, p1])


def laplas_function(x):
    return 0.5 * (1 + erf(x / np.sqrt(2)))


if __name__ == '__main__':
    result_1 = transform_matrices(E_LETTER, 1)
    result_2 = transform_matrices(O_LETTER, 1)

    figure = plt.figure(figsize=(10, 10))
    plt.title("Letters")
    sub_figure_1 = figure.add_subplot(2, 2, 1)
    plt.imshow(1 - E_LETTER, cmap='gray')
    sub_figure_1.set_title("'Е' letter")

    sub_figure_3 = figure.add_subplot(2, 2, 3)
    plt.imshow(1 - O_LETTER, cmap='gray')
    sub_figure_3.set_title("'P' letter")

    sub_figure_2 = figure.add_subplot(2, 2, 2)
    plt.imshow(1 - result_1, cmap='gray')
    sub_figure_2.set_title("Processed 'Е' letter")

    sub_figure_4 = figure.add_subplot(2, 2, 4)
    plt.imshow(1 - result_2, cmap='gray')
    sub_figure_4.set_title("Processed 'P' letter")
    plt.show()

    test_data_class_e = transform_matrices(E_LETTER, SET_SIZE, P_change)
    test_data_class_o = transform_matrices(O_LETTER, SET_SIZE, P_change)

    cond_prob_array_class_e = calc_condition_probs(test_data_class_e)
    cond_prob_array_class_o = calc_condition_probs(test_data_class_o)

    sd0, sd1 = calc_binarySD(cond_prob_array_class_e, cond_prob_array_class_o)
    m0, m1 = calc_M(cond_prob_array_class_e, cond_prob_array_class_o)

    print("M0: ", m0)
    print("M1: ", m1)
    print("SD0: ", sd0)
    print("SD1: ", sd1)

    fig = plt.figure
    plt.title("Плотность вероятностей X0(с) X1(к)")
    x0 = np.arange(-25, 25, 0.001)
    plt.plot(x0, norm.pdf(x0, m0, sd0), color='green', linewidth=3)

    x1 = np.arange(-25, 25, 0.001)
    plt.plot(x1, norm.pdf(x1, m1, sd1), color='blue', linewidth=3)

    lambda_tilda = calc_small_lambda(0.5, 0.5, cond_prob_array_class_e, cond_prob_array_class_o)
    array_lambda_tilda = np.zeros(4) + lambda_tilda
    plt.plot(array_lambda_tilda, np.arange(0, 0.2, 0.05), color='purple')
    print("lambda_tilda: ", lambda_tilda)
    plt.show()

    classified_array_class_e = classify_vectors_array(test_data_class_e,
                                                      0.5, 0.5,
                                                      cond_prob_array_class_e,
                                                      cond_prob_array_class_o)

    classified_array_class_o = classify_vectors_array(test_data_class_o,
                                                      0.5, 0.5,
                                                      cond_prob_array_class_o,
                                                      cond_prob_array_class_e)

    class_e_exp_error = calculate_exp_error(classified_array_class_e)
    class_o_exp_error = calculate_exp_error(classified_array_class_o)

    print("Экспериментальная ошибка классификации для класса Е:", class_e_exp_error)
    print("Экспериментальная ошибка классификации для класса О:", class_o_exp_error)

    theoretical_error = calc_theoretical_error(0.5, 0.5, cond_prob_array_class_e, cond_prob_array_class_o)

    print("Теоритическая ошибка классификации для класса Е:", theoretical_error[0])
    print("Теоритическая ошибка классификации для класса О:", theoretical_error[1])
