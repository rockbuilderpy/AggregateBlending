function Result({ result }) {
  return (
    <>
      {result.map((e, i) => {
        return <h3>{e.join(", ")}</h3>;
      })}
    </>
  );
}

export default Result;
