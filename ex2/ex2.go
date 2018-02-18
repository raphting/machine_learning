package ex2

import (
	"fmt"
	"github.com/cdipaolo/goml/linear"
	"github.com/cdipaolo/goml/base"
)

func Ex2() {
	exams()
	chips()
}

func exams() {
	x, y, err := base.LoadDataFromCSV("src/github.com/raphting/machine_learning/data/ex2data1.txt")
	if err != nil {
		fmt.Println(err.Error())
	}

	logistic := linear.NewLogistic(base.BatchGA, 0.001, 0,800, x, y)
	err = logistic.Learn()
	if err != nil {
		fmt.Println(err.Error())
	}


	// Prediction
	pred := []float64{45, 85}
	p, err := logistic.Predict(pred)
	if err != nil {
		fmt.Println(err.Error())
	}
	fmt.Printf("%f\n", p[0])
}

func chips() {
	x, y, err := base.LoadDataFromCSV("src/github.com/raphting/machine_learning/data/ex2data2.txt")
	if err != nil {
		fmt.Println(err.Error())
	}

	logistic := linear.NewLogistic(base.BatchGA, 0.001, 0, 800, x, y)
	err = logistic.Learn()
	if err != nil {
		fmt.Println(err.Error())
	}


	// Prediction
	pred := []float64{-0.25, 1.5}

	p, err := logistic.Predict(pred)
	if err != nil {
		fmt.Println(err.Error())
	}
	fmt.Printf("%f\n", p[0])
}
