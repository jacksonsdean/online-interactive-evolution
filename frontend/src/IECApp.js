import './IECApp.css';
import PopulationGrid from './components/populationGrid';
function IECApp() {
  return (
    <div className="IECApp">
      <header className="IECApp-header">
        <p>
          Interactive Evolutionary Art
        </p>
        <PopulationGrid />

      </header>
    </div>
  );
}

export default IECApp;
