import React from 'react';
import { API_URL } from '../Constants';
import { post } from '../util';

class IndividualButton extends React.Component {
    render() {
        return (
            <button className="individual-button" onClick={this.props.onClick}>
                {/* <img src={this.props.image} alt={this.props.alt} /> */}
                {this.props.name}
            </button>
        )
    }
}

class PopulationGrid extends React.Component {

    constructor(props) {
        super(props);
        this.state = { population_ids: ["Please wait.."] };
    }

    componentDidMount() {
        const apiUrl = API_URL;
        console.log(process.env.REACT_APP_NODE_ENV);
        console.log(apiUrl)
        post(apiUrl, { ids: [1, 2, 3, 4] })
            .then((response) => {
                if (response.status === 200 ){
                    return response.json()
                }
                else{
                    console.log("Error: " + response.status);
                    return Promise.reject(response.status);
                }
            }
            )
            .then((data) => {
                if ("body" in data) {
                    data =JSON.parse(data["body"]);
                }
                console.log(data);
                const ids = JSON.parse(data)
                this.setState({ population_ids: ids });
            })
    }

    render() {
        // list of all the population ids
        return (<div>
            <ul>
                {this.state.population_ids.map(i => <li key={i}> <IndividualButton name={i}></IndividualButton> </li>)}
            </ul>
        </div>
        );
    }
}
export default PopulationGrid;