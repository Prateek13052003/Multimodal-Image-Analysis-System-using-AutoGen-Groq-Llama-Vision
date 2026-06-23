import { useState } from "react";

export default function Home() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];

    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select an image first");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);

    const response = await fetch(
      "http://localhost:8000/api/analyze-image",
      {
        method: "POST",
        body: formData,
      }
    );

    const data = await response.json();

    setResult(data.result);
    setLoading(false);
  };

  return (
    <div className="container">
      <h1>Multimodal Image Analysis System</h1>

      <div className="upload-card">
        <input type="file" onChange={handleFileChange} />
      </div>

      {preview && (
        <div className="image-card">
          <img src={preview} alt="preview" className="preview-image" />
        </div>
      )}

      <button className="analyze-btn" onClick={handleUpload}>
        {loading ? "Analyzing..." : "Analyze Image"}
      </button>

      {result && (
        <div className="result-card">
          <h2>Analysis Results</h2>

          <p>
            <strong>People:</strong> {result.number_of_people}
          </p>

          <p>
            <strong>Scene:</strong> {result.scene}
          </p>

          <p>
            <strong>Activity:</strong> {result.activity}
          </p>

          <p>
            <strong>Emotion:</strong> {result.emotion}
          </p>

          <p>
            <strong>Category:</strong> {result.category}
          </p>

          <p>
            <strong>Image Type:</strong> {result.image_type}
          </p>

          <p>
            <strong>Colors:</strong>{" "}
            {result.dominant_colors.join(", ")}
          </p>
        </div>
      )}
    </div>
  );
}