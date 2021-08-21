function Input1({
  setSieveGradation,
  setNumberOfSieves,
  setNumberOfStocks,
  handleSubmit,
}) {
  return (
    <div className="input-container-1">
      <span>
        Sieve Gradation &nbsp;
        <select
          name="sieveGradation"
          onChange={(e) => setSieveGradation(e.target.value)}
        >
          <option value="WMM">WMM</option>
          <option value="DBM-37.5mm">DBM-37.5mm</option>
          <option value="DBM-26.5mm">DBM-26.5mm</option>
          <option value="BC-19mm">BC-19mm</option>
          <option value="BC-13.2mm">BC-13.2mm</option>
          <option value="SMA-13mm">SMA-13mm</option>
          <option value="SMA-19mm">SMA-19mm</option>
          <option value="DLC">DLC</option>
          <option value="PQC">PQC</option>
          <option value="Other">Other</option>
        </select>
      </span>
      <span>
        Number of sieves &nbsp;
        <input
          type="number"
          name="numberOfSieves"
          onChange={(e) => setNumberOfSieves(parseInt(e.target.value))}
        />
      </span>
      <span>
        Number of stocks &nbsp;
        <select
          name="numberOfStocks"
          onChange={(e) => setNumberOfStocks(parseInt(e.target.value))}
        >
          <option value="2">2</option>
          <option value="3">3</option>
          <option value="4">4</option>
          <option value="5">5</option>
        </select>
      </span>
      <input
        className="submit-btn"
        type="button"
        value="Submit"
        onClick={handleSubmit}
      />
    </div>
  );
}

export default Input1;
