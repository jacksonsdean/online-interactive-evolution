import React from 'react';
import {API_URL} from '../Constants';
import {string_to_ids, ids_to_string, post} from '../util';

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
        // interactive-evolutionary-computation?ids=1,2,3,4,5,10
        const apiUrl = API_URL + "?ids="+ids_to_string([1, 2, 3]);
        post(apiUrl, {"TEST_POST": "TEST_POST"})
            .then((response) => response.json())
            .then((data) => this.setState({ population_ids: string_to_ids(data) }));
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