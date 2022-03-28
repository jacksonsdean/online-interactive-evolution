import logo from './logo.svg';
import './App.css';
import WebRequestComponent from './components/webRequestComponent';
function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Welcome
        </p>
        <WebRequestComponent />

      </header>
    </div>
  );
}

export default App;
