from numpy import *


def cost(b, m, data_points):
    r_squared = 0

    # predicted Y^ = (mX + b)
    # Mean-Squared-Error = 1/2N(sum_for_N_Samples_of(y - y^)^2)
    for i in range(len(data_points)):
        X = data_points[i, 0]
        y = data_points[i, 1]
        r_squared += (y - (m * X + b)) ** 2
    return r_squared / (2 * float(len(data_points)))


def gradient_step(b, m, data_points, alpha): # Calculates Gradient for 1 iteration.
    b_grad = 0
    m_grad = 0
    N = float(len(data_points))
    for i in range(len(data_points)):
        X = data_points[i, 0]
        y = data_points[i, 1]
        b_grad += -(1/N) * (y - ((m * X) + b)) # partial derivative of Cost with respect to slope - 'b'.
        m_grad += -(1/N) * (X * (y - ((m * X) + b))) # partial derivative of Cost with respect to slope - 'm'.
    new_b = b - (alpha * b_grad)
    new_m = m - (alpha * m_grad)
    return [new_b, new_m]


def batch_gradient_descent(data_points, initial_b, initial_m, alpha, iterations):
    b = initial_b
    m = initial_m
    for i in range(iterations): # Calcultes Gradient Descent for n-iterations.
        b, m = gradient_step(b, m, array(data_points), alpha)
    return [b, m]


def predict_y(X):
    b, m = main()  # Gets the final slope and y-intercept after fitting the data.
    y_pred = m * X + b # Calculates y with converged 'm' and 'b'.
    print(f"The regressor prediction for input 'X' = {X} is 'y' = {y_pred}")


def main():
    data_points = genfromtxt("data.csv", delimiter=",") # Reads the data sheet.

    alpha = 0.0001  # Learning Rate

    # predicted Y = mX + b
    # m - slope, b - y_intercept
    initial_m = 0
    initial_b = 0
    iterations = 1000

    start_cost = cost(initial_b, initial_m, data_points) # Computes Cost before Gradient Descent.
    print(
        f"Starting Gradient Descent at 'y-intercept' = {initial_b}, 'slope' = {initial_m}, 'cost' = {start_cost}")

    [b, m] = batch_gradient_descent(
        data_points, initial_b, initial_m, alpha, iterations)

    end_cost = cost(b, m, data_points) # Computes cost after Gradient Descent.
    print(
        f"After performing Gradient Descent for {iterations} iterations we got: 'y-intercept' = {b}, 'slope' = {m}, 'cost' = {end_cost}")

    return b, m # return the slope and y-intercept after the Gradient Descent, in order to predict for the new values.


if __name__ == "__main__": # This is used to call the imported packages,it saves the memory.
    main()
    # predict_y(59.813207869512318) # original_y: 87.230925133687393,predicted_y: 88.48279754569299
