package vectors_test

import (
	"testing"

	"github.com/aws-samples/gofast-hnsw/vectordb/vectors"
	"github.com/stretchr/testify/assert"
)

func Test_GenerateRandomVectors(t *testing.T) {

	v, err := vectors.GenerateRandomVectors(8, 32)

	assert.Nil(t, err)

	assert.Equal(t, 8, len(v))

	assert.Equal(t, 32, len(v[0]))

	assert.LessOrEqual(t, v[0][0], float32(1.0))
	assert.GreaterOrEqual(t, v[1][0], float32(0.0))

}
