import React, { useState } from "react";
import axios from "axios";
import { Button, Container, Row, Col, Card, Form, Spinner } from "react-bootstrap";
import { PieChart, Pie, Cell, Tooltip, LineChart, Line, XAxis, YAxis, CartesianGrid, Legend } from "recharts";

const COLORS = ["#0088FE", "#00C49F", "#FFBB28"];

const App = () => {
    const [text, setText] = useState("");
    const [sentiment, setSentiment] = useState("");
    const [hashtags, setHashtags] = useState("");
    const [cluster, setCluster] = useState("");
    const [engagement, setEngagement] = useState(null);
    const [likes, setLikes] = useState(0);
    const [shares, setShares] = useState(0);
    const [comments, setComments] = useState(0);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const analyzeSentiment = async () => {
      setLoading(true);
      setError("");
      setSentiment(""); // Clear previous sentiment result before new request
  
      try {
          const response = await axios.post("http://127.0.0.1:5000/predict_sentiment", { text });
  
          // Check if response is valid
          if (response.data && response.data.sentiment_label) {
              setSentiment(response.data.sentiment_label);
          } else {
              setError("Invalid response from server.");
          }
  
          setText(""); // Clear input after successful response
  
      } catch (error) {
          console.error("API Error:", error);
          setError("Error analyzing sentiment. Please check your network or backend.");
      } finally {
          setLoading(false); // Ensure loading is disabled even if there's an error
      }
  };
  
    // Function to cluster hashtags
    const clusterHashtags = async () => {
        setLoading(true);
        setError("");
        try {
            const response = await axios.post("http://127.0.0.1:5000/predict_hashtag_cluster", { hashtags });
            setCluster(`Cluster ${response.data.cluster}`);
            setHashtags(""); // Clear input after submission
        } catch (error) {
            setError("Error clustering hashtags. Please try again.");
        }
        setLoading(false);
    };

    // Function to predict engagement
    const predictEngagement = async () => {
        setLoading(true);
        setError("");
        try {
            const response = await axios.post("http://127.0.0.1:5000/predict_engagement", { likes, shares, comments });
            setEngagement(response.data.predicted_engagement);
        } catch (error) {
            setError("Error predicting engagement. Please try again.");
        }
        setLoading(false);
    };

    return (
        <Container className="mt-4">
            <h2 className="text-center">📊 Social Media Analytics Dashboard</h2>

            {error && <p className="text-danger text-center">{error}</p>}

            {/* Sentiment Analysis */}
            <Card className="p-3 mb-4">
                <h4>📢 Sentiment Analysis</h4>
                <Row>
                    <Col md={8}>
                        <Form.Control
                            type="text"
                            placeholder="Enter text for sentiment analysis"
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                        />
                    </Col>
                    <Col md={4}>
                        <Button variant="primary" onClick={analyzeSentiment} disabled={loading}>
                            {loading ? <Spinner animation="border" size="sm" /> : "Analyze Sentiment"}
                        </Button>
                    </Col>
                </Row>
                {sentiment && <p className="mt-2">Sentiment: <strong>{sentiment}</strong></p>}
            </Card>

            {/* Hashtag Clustering */}
            <Card className="p-3 mb-4">
                <h4>#️⃣ Hashtag Clustering</h4>
                <Row>
                    <Col md={8}>
                        <Form.Control
                            type="text"
                            placeholder="Enter hashtags"
                            value={hashtags}
                            onChange={(e) => setHashtags(e.target.value)}
                        />
                    </Col>
                    <Col md={4}>
                        <Button variant="success" onClick={clusterHashtags} disabled={loading}>
                            {loading ? <Spinner animation="border" size="sm" /> : "Cluster Hashtags"}
                        </Button>
                    </Col>
                </Row>
                {cluster && <p className="mt-2">Cluster: <strong>{cluster}</strong></p>}
            </Card>

            {/* Engagement Prediction */}
            <Card className="p-3 mb-4">
                <h4>📈 Engagement Prediction</h4>
                <Row>
                    <Col md={4}>
                        <Form.Control type="number" placeholder="Likes" value={likes} onChange={(e) => setLikes(e.target.value)} />
                    </Col>
                    <Col md={4}>
                        <Form.Control type="number" placeholder="Shares" value={shares} onChange={(e) => setShares(e.target.value)} />
                    </Col>
                    <Col md={4}>
                        <Form.Control type="number" placeholder="Comments" value={comments} onChange={(e) => setComments(e.target.value)} />
                    </Col>
                </Row>
                <Button variant="warning" className="mt-2" onClick={predictEngagement} disabled={loading}>
                    {loading ? <Spinner animation="border" size="sm" /> : "Predict Engagement"}
                </Button>
                {engagement && <p className="mt-2">Predicted Engagement: <strong>{engagement}</strong></p>}
            </Card>

            {/* Visualization Charts */}
            <Row>
                <Col md={6}>
                    <h5>📊 Sentiment Distribution</h5>
                    <PieChart width={300} height={300}>
                        <Pie
                            data={[
                                { name: "Positive", value: sentiment === "Positive" ? 1 : 0 },
                                { name: "Neutral", value: sentiment === "Neutral" ? 1 : 0 },
                                { name: "Negative", value: sentiment === "Negative" ? 1 : 0 },
                            ]}
                            cx="50%"
                            cy="50%"
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="value"
                            label
                        >
                            {COLORS.map((color, index) => (
                                <Cell key={`cell-${index}`} fill={color} />
                            ))}
                        </Pie>
                        <Tooltip />
                    </PieChart>
                </Col>

                <Col md={6}>
                    <h5>📈 Engagement Trends</h5>
                    <LineChart width={400} height={250} data={[{ name: "Engagement", value: engagement || 0 }]}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="value" stroke="#82ca9d" />
                    </LineChart>
                </Col>
            </Row>
        </Container>
    );
};

export default App;
