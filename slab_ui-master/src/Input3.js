function Input3({ numberOfSieves, numberOfStocks, handleInput3Change }) {
  return numberOfSieves === 0 ? null : (
    <table>
      <caption>Enter weights retained on sieves</caption>
      <thead>
        <tr>
          <td>Sieve Sizes</td>
          {[...Array(numberOfStocks)].map((e, i) => {
            return <td key={i}>Stock {i + 1}</td>;
          })}
        </tr>
      </thead>
      <tbody>
        {[...Array(numberOfSieves + 1)].map((e, i) => {
          return (
            <tr key={i}>
              {i === numberOfSieves ? <td>Pan</td> : <td>Sieve {i + 1}</td>}
              {[...Array(numberOfStocks)].map((ee, ii) => {
                return (
                  <td key={ii}>
                    <input
                      type="number"
                      onChange={(e) =>
                        handleInput3Change(i, ii, e.target.value)
                      }
                    />
                  </td>
                );
              })}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export default Input3;
