import React, { useEffect, useState } from 'react';
import axios from 'axios';

function App() {
  const [data, setData] = useState([]);

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/api/items/")
      
      .then(res => {
        setData(res.data);
        console.log(res.data); 
      })
      .catch(err => console.error(err));
  }, []);

  return (
    <div>
      <h1>Django + React + PostgreSQL</h1>
      <ul>
        {data.map((item, i) => <li key={i}>{item.name}</li>)}
      </ul>
    </div>
  );
}

export default App;
