import { Link } from "react-router-dom";
import { useState } from "react";
import Input1 from "./Input1";
import Input2 from "./Input2";
import Input3 from "./Input3";

function Inputs({ setResult }) {
  const [sieveGradation, setSieveGradation] = useState("WMM");
  const [numberOfSieves, setNumberOfSieves] = useState(0);
  const [numberOfStocks, setNumberOfStocks] = useState(0);
  const [noOfSieves, setNoOfSieves] = useState(0);
  const [noOfStocks, setNoOfStocks] = useState(0);
  const [input2, setInput2] = useState([]);
  const [input3, setInput3] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isError, setIsError] = useState(false);
  const [data, setData] = useState([]);

  const handleSubmit = () => {
    setNoOfSieves(numberOfSieves);
    setNoOfStocks(numberOfStocks);
    let newInput = Array(numberOfSieves);
    for (let i = 0; i < numberOfSieves; i++) {
      newInput[i] = Array(2);
    }
    setInput2(newInput);
    newInput = Array(numberOfSieves + 1);
    for (let i = 0; i < numberOfSieves + 1; i++) {
      newInput[i] = Array(numberOfStocks);
    }
    setInput3(newInput);
  };

  const handleInput2Change = (i, j, num) => {
    num = parseInt(num);
    setInput2((prevInput) => {
      let newInput = JSON.parse(JSON.stringify(prevInput));
      newInput[i][j] = num;
      return newInput;
    });
  };

  const handleInput3Change = (i, j, num) => {
    num = parseInt(num);
    setInput3((prevInput) => {
      let newInput = JSON.parse(JSON.stringify(prevInput));
      newInput[i][j] = num;
      return newInput;
    });
  };

  const calculate = async () => {
    setIsError(false);
    setIsLoading(true);
    const requestOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input2: input2, input3: input3 }),
    };
    const response = await fetch(
      "http://localhost:5000/calculate",
      requestOptions
    );
    const data = await response.json();
    if (data.status === "error") {
      setIsError(true);
      setData(data.description);
    } else {
      setResult(data.data);
      setData(data.data);
    }
    setIsLoading(false);
  };

  return (
    <>
      <Input1
        setSieveGradation={setSieveGradation}
        setNumberOfSieves={setNumberOfSieves}
        setNumberOfStocks={setNumberOfStocks}
        handleSubmit={handleSubmit}
      />
      <Input2
        numberOfSieves={noOfSieves}
        handleInput2Change={handleInput2Change}
      />
      <Input3
        numberOfSieves={noOfSieves}
        numberOfStocks={noOfStocks}
        handleInput3Change={handleInput3Change}
      />
      {noOfSieves > 0 && (
        <input
          className={isLoading ? "submit-btn disabled-btn" : "submit-btn"}
          type="button"
          value="Calculate"
          onClick={calculate}
          disabled={isLoading}
        />
      )}
      {isError ? (
        <h1>{data}</h1>
      ) : (
        data.length > 0 && (
          <button className={"submit-btn"}>
            <Link to="/result">View Results</Link>
          </button>
        )
      )}
    </>
  );
}

export default Inputs;
