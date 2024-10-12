package embeddings

import "fmt"

type Model string

const (
	Llama32              Model = "llama3.2"
	MxbaiEmbedLarge      Model = "mxbai-embed-large"
	SnowflakeArcticEmbed Model = "snowflake-arctic-embed"
)

func (m Model) Name() string {
	return string(m)
}

func ModelFromString(s string) (Model, error) {
	for _, model := range []Model{Llama32, MxbaiEmbedLarge, SnowflakeArcticEmbed} {
		if model.Name() == s {
			return model, nil
		}
	}
	return "", fmt.Errorf("Unknown model: %s", s)
}
