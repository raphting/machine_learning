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
	fmt.Printf("c = %v\n", fc)

	// Ex. 2
	population, profit := loadData()
	dimensions := 2
	persist := false
	debug := false
	plot, _ := glot.NewPlot(dimensions, persist, debug)
	plot.AddPointGroup("data", "points", [][]float64{population, profit})
	plot.SetXLabel("Population of City in 10,000s")
	plot.SetYLabel("Profit in $10,000s")
	plot.SavePlot("src/github.com/raphting/machine_learning/plot/ex1.png")

	m := len(profit)
	ones := make([]float64, m)
	for k := range ones {
		ones[k] = 1
	}


	// Create Feature Vector including ones
	X := mat.NewDense(m, 2, nil)
	X.SetCol(0, ones)
	X.SetCol(1, population)
	fc = mat.Formatted(X, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("c = %v\n", fc)

	iterations := 1500
	alpha := 0.01
	theta0 := float64(0)
	theta1 := float64(0)
	for i := 0; i < iterations; i++ {
		e0 := float64(0)
		for c := 0; c < m; c++ {
			e0 += (theta0 + (theta1 * population[c])) - profit[c]
		}
		tmp0 := theta0 - (alpha * (e0 / float64(m)))

		e1 := float64(0)
		for c := 0; c < m; c++ {
			e1 += (theta0 + (theta1 * population[c]) - profit[c]) * population[c]
		}
		tmp1 := theta1 - (alpha * (e1 / float64(m)))

		theta0 = tmp0
		theta1 = tmp1
	}

	fmt.Println(theta0, theta1)
	cost := float64(0)
	for c := 0; c < m; c++ {
		cost += math.Pow(((theta0 + (theta1 * population[c])) - profit[c]), 2)
	}
	cost = cost / (float64(2*m))
	fmt.Println("Cost: ", cost)


	fct := func(x float64) float64 { return (theta0 + theta1 * x) }
	groupName := "Linear Regression"
	style := "lines"
	pointsX := []float64{}
	for xx := float64(0); xx < 25; xx += 0.1 {
		pointsX = append(pointsX, xx)
	}
	plot.AddFunc2d(groupName, style, pointsX, fct)
	plot.SavePlot("src/github.com/raphting/machine_learning/plot/ex1.png")
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
