import React from 'react';
import { API_URL, INITIAL_REQUEST } from '../Constants';
import { post } from '../util';

const concat = (xs, ys) => xs.concat(ys);

class IndividualButton extends React.Component {
    render() {
        console.log(JSON.parse(this.props.individual.image))
        console.log(typeof(JSON.parse(this.props.individual.image)))
        // create image from individual
        const parsed = JSON.parse(this.props.individual.image)
        const bytesArray = btoa(String.fromCharCode.apply(null, parsed))
        console.log(bytesArray)
        return (
            <button className="individual-button" onClick={this.props.onClick}>
                {this.props.individual.name}
                {/* {this.props.image} */}
                <img src={"data:image/png;base64," + bytesArray } alt={this.props.individual.name} />
                {/* <img src={URL.createObjectURL(blob)} alt={this.props.individual.name} /> */}
            </button>
        )
    }
}

class PopulationGrid extends React.Component {

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
        return (<div>
            <ul>
                {this.state.population.map((obj, index)=> <li key={index}> <IndividualButton individual={obj}></IndividualButton> </li>)}
            </ul>
        </div>
        );
    }
}
export default PopulationGrid;