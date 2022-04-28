import React from 'react';
import { API_URL, INITIAL_REQUEST } from '../Constants';
import { post } from '../util';
import Grid  from './Grid';
import styles from "./PopulationGrid.module.css";

class IndividualButton extends React.Component {

    constructor(props){
        super(props)
        this.state = {individual:props.individual}
        this.clicked = this.clicked.bind(this)
    }

    clicked() {
        this.props.individual.selected = !this.props.individual.selected
        this.setState({individual:this.props.individual})
    }

    render() {
        const individual = this.props.individual;
        // create image from individual
        const parsed = JSON.parse(individual.image)
        // create url from base64 string
        const url = "data:image/png;base64,"+parsed.join("")
        console.log(individual.selected)
        const selectionStyle = individual.selected ? styles.selected:styles.unselected
        return (
            <button className={styles.individualButton + " " + selectionStyle} onClick={this.clicked}>
                {individual.name}
                <img className={styles.individualImg} src={url} alt={individual.name} />
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
        const apiUrl = API_URL;
        console.log(process.env.REACT_APP_NODE_ENV);
        console.log(apiUrl)
        post(apiUrl, INITIAL_REQUEST)
            .then((response) => {
                console.log(response)
                if (response.status === 200 ){
                    return response.json()
                }
                else{
                    console.log("Error: " + response.status);
                    return Promise.reject(response.status);
                }
            }
            ).catch((error) => {
                console.log("Error: " + error);})
                .then((data) => {
                    if ("body" in data && typeof data.body !== "object") {
                    data =JSON.parse(data["body"]);
                }
                console.log(data);
                const pop = data["population"];
                this.setState({ population: pop });
            })
            .catch((error) => {
                console.log("Error: " + error);})
    }

    render() {
        if (typeof(this.state.population) === 'undefined' || this.state.population.length === 0) {
            return (<p>Please wait...</p>)
        }
        // a grid of the population's individuals' images as buttons
        return (<div className={styles.populationGrid}>
            <Grid>
                {this.state.population.map((obj, index)=> <IndividualButton key={index} individual={obj}></IndividualButton>)}
            </Grid>
        </div>
        );
    }
}
export default PopulationGrid;