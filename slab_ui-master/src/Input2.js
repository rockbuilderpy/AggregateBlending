function Input2({ numberOfSieves, handleInput2Change }) {
  return numberOfSieves === 0 ? null : (
    <table>
      <caption>Enter bounds expressed as percentage passing</caption>
      <thead>
        <tr>
          <td>Sieve Sizes</td>
          <td>Lower Bound</td>
          <td>Upper Bound</td>
        </tr>
      </thead>
      <tbody>
        {[...Array(numberOfSieves)].map((e, i) => {
          return (
            <tr key={i}>
              <td>Sieve {i + 1}</td>
              <td>
                <input
                  type="number"
                  onChange={(e) => handleInput2Change(i, 0, e.target.value)}
                />
              </td>
              <td>
                <input
                  type="number"
                  onChange={(e) => handleInput2Change(i, 1, e.target.value)}
                />
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export default Input2;
