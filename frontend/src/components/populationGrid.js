import React from 'react';
import { DEFAULT_CONFIG } from '../Constants';
import { initialPopulation, nextGeneration } from '../util';
import Grid from './Grid';
import styles from "./PopulationGrid.module.css";

class IndividualButton extends React.Component {

    constructor(props) {
        super(props)
        this.state = { individual: props.individual }
        this.clicked = this.clicked.bind(this)
    }

    clicked() {
        this.props.individual.selected = !this.props.individual.selected
        this.setState({ individual: this.props.individual })
    }

    render() {
        const individual = this.props.individual;
        // create image from individual
        const parsed = JSON.parse(individual.image)
        // create url from base64 string
        const url = "data:image/png;base64," + parsed.join("")
        const selectionStyle = individual.selected ? styles.selected : styles.unselected
        return (
            <button className={styles.individualButton + " " + selectionStyle} onClick={this.clicked}>
                <img className={styles.individualImg} src={url} alt={individual.name} />
            </button>
        )
    }
}

class NextGenerationButton extends React.Component {

    constructor(props) {
        super(props)
    }

    render() {
        return (
            <button className={styles.nextGenButton} onClick={this.props.onClick}>
                Next Generation
            </button>
        )
    }
}

class PopulationGrid extends Grid {

    constructor(props) {
        super(props);
        this.state = { population: [] };
    }

    componentDidMount() {
        initialPopulation(DEFAULT_CONFIG)
            .then((data) => {
                if ("body" in data && typeof data.body !== "object") {
                    data = JSON.parse(data["body"]);
                }
                console.log(data);
                const pop = data["population"];
                console.log(pop);
                this.setState({ population: pop });
                console.log(this.state.population);
            })
    }

    render() {
        if (typeof (this.state.population) === 'undefined' || this.state.population.length === 0) {
            return (<p>Please wait...</p>)
        }
        // a grid of the population's individuals' images as buttons
        return (<><div className={styles.populationGrid}>
            <Grid>
                {this.state.population.map((obj, index) => <IndividualButton key={index} individual={obj}></IndividualButton>)}
            </Grid>
        </div>
            <NextGenerationButton onClick={() => {
                nextGeneration(this.state.population, DEFAULT_CONFIG)
            }}></NextGenerationButton>
        </>
        );
    }
}
export default PopulationGrid;