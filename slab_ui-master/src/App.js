import { useState } from "react";
import { BrowserRouter as Router, Route } from "react-router-dom";
import "./App.css";
import Inputs from "./Inputs";
import Result from "./Result";

function App() {
  const [result, setResult] = useState([]);

  return (
    <Router>
      <div className="App">
        <Route
          path="/"
          exact={true}
          render={(props) => <Inputs setResult={setResult} {...props} />}
        />
        <Route
          path="/result"
          render={(props) => <Result result={result} {...props} />}
        />
      </div>
    </Router>
  );
}

export default App;
