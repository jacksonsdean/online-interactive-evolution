import './IECApp.css';
import WebRequestComponent from './components/webRequestComponent';
function IECApp() {
  return (
    <div className="IECApp">
      <header className="IECApp-header">
        <p>
          Welcome
        </p>
        <WebRequestComponent />

      </header>
    </div>
  );
}

export default IECApp;
