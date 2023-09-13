package distance

import (
	"errors"
)

func L2(queryPoint []float32, vectorToCompare []float32) (distance float32, err error) {

	qplen := len(queryPoint)

	if qplen != len(vectorToCompare) {
		return 0, errors.New("Must compare two vectors of the same dimension")
	}

	return L2_1x(queryPoint, vectorToCompare)

	// Should not reach here
	return 0, nil

}

/*
func L2_Opt(queryPoint *[]float32, vectorToCompare *[]float32) (distance float32, err error) {

	qplen := len(*queryPoint)

	if qplen != len(*vectorToCompare) {
		return 0, errors.New("Must compare two vectors of the same dimension")
	}

	return L2_1x_Opt(queryPoint, vectorToCompare), nil

	// Should not reach here
	return 0, nil

}
*/

func L2_1x(queryPoint []float32, vectorToCompare []float32) (distance float32, err error) {

	for i := 0; i < len(queryPoint); i++ {
		distance += (queryPoint[i] - vectorToCompare[i]) * (queryPoint[i] - vectorToCompare[i])
	}

	return distance, nil
}

func L2_Opt(queryPoint *[]float32, vectorToCompare *[]float32) (distance float32, err error) {

	//if len(*queryPoint) != len(*vectorToCompare) {
	//	return 0, errors.New("Must compare two vectors of the same dimension")
	//}

	for i := 0; i < len(*queryPoint); i++ {
		distance += ((*queryPoint)[i] - (*vectorToCompare)[i]) * ((*queryPoint)[i] - (*vectorToCompare)[i])
	}

	return distance, nil
}

// TODO: Benchmark performance, does the go compiler optimise this?! cc flags? SEEMS NO DIFFERENCE!
/*
func L2_4x(queryPoint []float32, vectorToCompare []float32) (distance float32, err error) {

	qty := len(queryPoint) >> 2
	fmt.Println("L2_4x qty ", len(queryPoint), qty)

	var b = 1
	for i := 0; i < qty; i++ {
		idx := i * b

		distance += (queryPoint[idx] - vectorToCompare[idx]) * (queryPoint[idx] - vectorToCompare[idx])

		distance += (queryPoint[idx+1] - vectorToCompare[idx+1]) * (queryPoint[idx+1] - vectorToCompare[idx+1])

		distance += (queryPoint[idx+2] - vectorToCompare[idx+2]) * (queryPoint[idx+2] - vectorToCompare[idx+2])

		distance += (queryPoint[idx+3] - vectorToCompare[idx+3]) * (queryPoint[idx+3] - vectorToCompare[idx+3])
		b++

	}

	return distance, nil

}

// Pointer version

func L2_4x_Opt(queryPoint *[]float32, vectorToCompare *[]float32) (distance float32) {

	//qty := len(queryPoint)

	qty := len(*queryPoint) >> 2

	for i := 0; i < qty; i++ {
		distance += ((*queryPoint)[i] - (*vectorToCompare)[i]) * ((*queryPoint)[i] - (*vectorToCompare)[i])

		distance += ((*queryPoint)[i+1] - (*vectorToCompare)[i+1]) * ((*queryPoint)[i+1] - (*vectorToCompare)[i+1])

		distance += ((*queryPoint)[i+2] - (*vectorToCompare)[i+2]) * ((*queryPoint)[i+2] - (*vectorToCompare)[i+2])

		distance += ((*queryPoint)[i+3] - (*vectorToCompare)[i+3]) * ((*queryPoint)[i+3] - (*vectorToCompare)[i+3])

	}

	return distance

}
*/
