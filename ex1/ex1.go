package main

import (
	"gonum.org/v1/gonum/mat"
	"fmt"
	"io/ioutil"
	"strings"
	"strconv"
	"github.com/Arafatk/glot"
	"math"
)

func main() {
	// Ex. 1
	d := []float64{1, 1, 1, 1, 1}
	A := mat.NewDiagonal(5, d)
	fc := mat.Formatted(A.T(), mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("A = %v\n", fc)

	// Ex. 2.1
	population, profit := loadData()

	// Ex. 2.2
	m := len(profit)
	ones := make([]float64, m)
	for k := range ones {
		ones[k] = 1
		}


	// Create Feature Vector including ones
	X := mat.NewDense(m, 2, nil)
	X.SetCol(0, ones)
	X.SetCol(1, population)
	// Create trained data
	y := mat.NewDense(m, 1, profit)
	// Create start values for theta
	theta := mat.NewDense(2, 1, []float64{0, 0})

	// Create dummy cost for tests (should be ~32.07)
	cost := calcCost(X, y, theta)
	fmt.Println("Cost: ", cost)

	iterations := 1500
	alpha := 0.01
	theta = gradientDescent(X, y, theta, alpha, iterations)
	plotData(population, profit, theta, "src/github.com/raphting/machine_learning/plot/ex1.png")

	// Predictions
	pred1 := mat.NewDense(1, 2, []float64{1, 3.5})
	pred2 := mat.NewDense(1, 2, []float64{1, 7})

	res := mat.NewDense(1, 1, nil)
	res.Mul(pred1, theta)
	fc = mat.Formatted(res, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("Prediction1 = %v\n", fc)

	res.Mul(pred2, theta)
	fc = mat.Formatted(res, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("Prediction2 = %v\n", fc)

	optionalTasks()
}

func plotData(x, y []float64, theta *mat.Dense, path string) {
	dimensions := 2
	persist := false
	debug := false
	plot, _ := glot.NewPlot(dimensions, persist, debug)
	plot.AddPointGroup("data", "points", [][]float64{x, y})
	plot.SetXLabel("Population of City in 10,000s")
	plot.SetYLabel("Profit in $10,000s")

	theta0 := theta.At(0,0)
	theta1 := theta.At(1, 0)
	fct := func(x float64) float64 { return (theta0 + theta1 * x) }
	groupName := "Linear Regression"
	style := "lines"
	pointsX := []float64{}
	for xx := float64(3); xx < 25; xx += 0.1 {
		pointsX = append(pointsX, xx)
	}
	plot.AddFunc2d(groupName, style, pointsX, fct)
	plot.SavePlot(path)
}

func calcCost(X, y, theta *mat.Dense) float64 {
	m, _ := y.Dims()
	cost := float64(0)
	for i := 0; i < m; i++ {
		J := mat.NewDense(1, 1, []float64{cost})
		J.Mul(theta.T(), X.RowView(i))
		cost += math.Pow(J.At(0,0) - y.At(i, 0), 2)
	}
	return cost / float64(2*m)
}

func loadData() ([]float64, []float64) {
	d, err := ioutil.ReadFile("src/github.com/raphting/machine_learning/data/ex1data1.txt")
	if err != nil {
		fmt.Println(err.Error())
	}

	data := string(d)
	split := strings.Split(data, "\n")
	split = split[:len(split)-1] // ACHTUNG!!!

	population := make([]float64, len(split))
	profit := make([]float64, len(split))
	for k, s := range split {
		tmp := strings.Split(s, ",")
		i, err := strconv.ParseFloat(tmp[0], 64)
		if err != nil {
			fmt.Println(err.Error())
			population[k] = 0
			profit[k] = 0
			continue
		}
		population[k] = float64(i)

		i, err = strconv.ParseFloat(tmp[1], 64)
		if err != nil {
			fmt.Println(err.Error())
			profit[k] = 0
			continue
		}
		profit[k] = float64(i)
	}

	return population, profit
}

func gradientDescent(X, y, theta *mat.Dense, alpha float64, iters int) *mat.Dense {
	m, _ := y.Dims()
	rate := alpha / float64(m)

	for i := 0; i < iters; i++ {
		theta = calcTheta(rate, X, y, theta)
	}
	return theta
}

func calcTheta(rate float64, X, y, theta *mat.Dense) *mat.Dense {
	m, _ := y.Dims()
	t, _ := theta.Dims()

	cost := make([]float64, t)
	thetas := make([]float64, t)
	for c := range cost {
		cost[c] = 0
		thetas[c] = theta.At(c, 0)
	}

	for i := 0; i < m; i++ {
		J := mat.NewDense(1, 1, nil)
		J.Mul(theta.T(), X.RowView(i)) // a + bx
		h := J.At(0, 0)

		for c := range cost {
			cost[c] += (h - y.At(i, 0)) * X.At(i, c) // E (h(x) - y) * x
		}
	}

	for t := range thetas {
		thetas[t] = thetas[t] - rate * cost[t]
	}

	return mat.NewDense(t, 1, thetas)
}
