package embeddings

type Document struct {
	FileId    string    `json:"file_id"`
	Embedding []float64 `json:"embedding"`
	Text      string    `json:"text"`
}
