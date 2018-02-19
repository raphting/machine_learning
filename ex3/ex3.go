package ex3

import (
	"fmt"
	"strings"
	"io/ioutil"
	"strconv"
	"github.com/cdipaolo/goml/linear"
	"github.com/cdipaolo/goml/base"
	"time"
	"math"
)

func Ex3() {
	x, y := loadData()

	// Prepare one-vs-all
	class := make([][]float64, 10)
	for c := range class {
		class[c] = make([]float64, len(y))
	}

	for yy := range y {
		thisclass := int(y[yy])
		if thisclass == 10 {
			thisclass = 0
		}
		class[thisclass][yy] = 1
	}
	
	// Learn
	alpha := 0.001
	regul := float64(6)
	maxIter := 1000

	// Prepare all logistic classes
	logistic := make([]*linear.Logistic, 10)
	for i := 0; i < 10; i++ {
		logistic[i] = linear.NewLogistic(base.BatchGA, alpha, regul, maxIter, x, class[i])
	}

	// Learn them in parallel (takes about 1h on an i5)
	lErr := make(chan error, 10)
	for i := 0; i < 10; i++ {
		go func(i int) {lErr <- logistic[i].Learn()}(i)
	}

	// Progress loop. Exists after all learners exit with err == nil
	for {
		if len(lErr) == 10 {
			break
		}

		fmt.Println("Sum of already learned: ", len(lErr))
		time.Sleep(10 * time.Second)
	}

	// Predicting (worked in the test. Zero = 1, Two = 0.00000002, Rest = 0)
	for i := 0; i < 10; i++ {
		fmt.Println(logistic[i].Predict(x[0]))
	}
}

func Ex3_nn() {
	x, y := loadData()
	th1, th2 := loadTheta()

	correct := make([]int, 10)
	wrong := make([]int, 10)
	for cntX := range x {
		predX := x[cntX]

		// Predict
		// Bias = 1
		predX = append([]float64{1}, predX...)
		hidden1 := make([]float64, 25)
		// 1st hidden layer
		for h := range hidden1 {
			for xx := range predX {
				hidden1[h] += predX[xx] * th1[h][xx]
			}
			hidden1[h] = sigmoid(hidden1[h])
		}

		// Bias output = 1
		hidden1 = append([]float64{1}, hidden1...)
		output := make([]float64, 10)
		// Output layer
		for o := range output {
			for h := range hidden1 {
				output[o] += hidden1[h] * th2[o][h]
			}
		}

		biggest := output[0]
		resNum := 0
		for k, v := range output {
			if v > biggest {
				biggest = v
				resNum = k
			}
		}

		expected := int(y[cntX])
		if resNum + 1 == expected {
			correct[expected-1]++
			continue
		}
		wrong[expected-1]++
	}

	fmt.Println("Correct: ", correct, " Wrong: ", wrong)
}

func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

func loadData() ([][]float64, []float64) {
	d, err := ioutil.ReadFile("src/github.com/raphting/machine_learning/data/ex3data1_X.txt")
	if err != nil {
		fmt.Println(err.Error())
	}

	data := string(d)
	split := strings.Split(data, "\n")
	split = split[:len(split)-1]

	x := make([][]float64, len(split))
	for k, s := range split {
		tmp := strings.Split(s, " ")
		tmp = tmp[1:]

		row := make([]float64, len(tmp))
		for t := range tmp {
			i, _ := strconv.ParseFloat(tmp[t], 64)
			row[t] = i
		}
		x[k] = row
	}

	fmt.Println("X is ", len(x))

	d, err = ioutil.ReadFile("src/github.com/raphting/machine_learning/data/ex3data1_y.txt")
	if err != nil {
		fmt.Println(err.Error())
	}

	data = string(d)
	split = strings.Split(data, "\n")
	split = split[:len(split)-1]

	y := make([]float64, len(split))
	for k, s := range split {
		i, _ := strconv.ParseFloat(s, 64)
		y[k] = i
	}
	fmt.Println("Y is ", len(y))

	return x, y
}

func loadTheta() ([][]float64, [][]float64) {
	d, err := ioutil.ReadFile("src/github.com/raphting/machine_learning/data/ex3data1_Th1.txt")
	if err != nil {
		fmt.Println(err.Error())
	}

	data := string(d)
	split := strings.Split(data, "\n")

	th1 := make([][]float64, len(split))
	for k, s := range split {
		tmp := strings.Split(s, " ")
		tmp = tmp[1:]

		row := make([]float64, len(tmp))
		for t := range tmp {
			i, _ := strconv.ParseFloat(tmp[t], 64)
			row[t] = i
		}
		th1[k] = row
	}

	d, err = ioutil.ReadFile("src/github.com/raphting/machine_learning/data/ex3data1_Th2.txt")
	if err != nil {
		fmt.Println(err.Error())
	}

	data = string(d)
	split = strings.Split(data, "\n")

	th2 := make([][]float64, len(split))
	for k, s := range split {
		tmp := strings.Split(s, " ")
		tmp = tmp[1:]

		row := make([]float64, len(tmp))
		for t := range tmp {
			i, _ := strconv.ParseFloat(tmp[t], 64)
			row[t] = i
		}
		th2[k] = row
	}

	fmt.Println("Th1 is ", len(th1), " X ", len(th1[0]))
	fmt.Println("Th2 is ", len(th2), " X ", len(th2[0]))
	return th1, th2
}
