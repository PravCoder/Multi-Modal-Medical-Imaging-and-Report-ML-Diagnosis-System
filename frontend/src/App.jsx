import React, { useEffect, useState } from "react";
import axios from "axios";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import HomePage from "./pages/HomePage.jsx"

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
      {/* <h1>Django + React + PostgreSQL</h1>
      <ul>
        {data.map((item, i) => <li key={i}>{item.name}</li>)}
      </ul> */}
      <BrowserRouter>
        <Routes>

          <Route path="/" element={<HomePage />}/>


        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
