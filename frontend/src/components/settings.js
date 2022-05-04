import { DEFAULT_CONFIG } from "Constants";
import React from "react";
import ReactSlider from 'react-slider'
import styles from "./Settings.module.css"
import styled from "styled-components";

import { ProSidebar, Menu, MenuItem, SubMenu } from 'react-pro-sidebar';
import 'react-pro-sidebar/dist/css/styles.css';


const StyledProSidebar = styled(ProSidebar)`
position: fixed;
top: 0;
right:0;
height:fit-content;
width: 400px;
white-space: pre;
`;

const StyledMenu = styled(Menu)`
background-color: rgba(200,200,255,.1);
`;

const StyledDropDown = styled.select`
background-color: rgba(200,200,255,.1);
color:white;
margin-left:10px;
`;


const StyledTrack = styled.div`
  top: 0;
  bottom: 0;
  background: ${props =>
        props.index === 2 ? "#f00" : props.index === 1 ? "#0f0" : "#ddd"};
  border-radius: 999px;
`;

const Track = (props, state) => <StyledTrack {...props} index={state.index} />;


function Settings() {
    return (
        <form>
            <StyledProSidebar>
                <StyledMenu iconShape="square">
                    <SubMenu title="Settings">
                        <SubMenu title="General">
                            <MenuItem>
                                Radial symmetry bias<input type="checkbox" />
                            </MenuItem>
                            <MenuItem>
                                {"Color mode"}
                                <StyledDropDown>
                                    <option value="L">Grayscale</option>
                                    <option value="HSL">HSL</option>
                                    <option selected value="RGB">RGB</option>
                                </StyledDropDown>
                            </MenuItem>
                        </SubMenu>

                        <SubMenu title="Mutation rates">
                            <MenuItem>Weights
                                <ReactSlider
                                    min={0}
                                    max={1}
                                    step={.01}
                                    className={styles.horizontalSlider}
                                    thumbClassName={styles.thumb}
                                    trackClassName={styles.track}
                                    renderThumb={(props, state) => <div {...props}>{state.valueNow}</div>}
                                />
                            </MenuItem>
                            <MenuItem>Add nodes
                                <ReactSlider
                                    min={0}
                                    max={1}
                                    step={.01}
                                    className={styles.horizontalSlider}
                                    thumbClassName={styles.thumb}
                                    trackClassName={styles.track}
                                    renderThumb={(props, state) => <div {...props}>{state.valueNow}</div>}
                                />
                            </MenuItem>
                            <MenuItem>Remove nodes
                                <ReactSlider
                                    min={0}
                                    max={1}
                                    step={.01}
                                    className={styles.horizontalSlider}
                                    thumbClassName={styles.thumb}
                                    trackClassName={styles.track}
                                    renderThumb={(props, state) => <div {...props}>{state.valueNow}</div>}
                                />
                            </MenuItem>
                            <MenuItem>Add connections
                                <ReactSlider
                                    min={0}
                                    max={1}
                                    step={.01}
                                    className={styles.horizontalSlider}
                                    thumbClassName={styles.thumb}
                                    trackClassName={styles.track}
                                    renderThumb={(props, state) => <div {...props}>{state.valueNow}</div>}
                                />
                            </MenuItem>
                            <MenuItem>Remove connections
                                <ReactSlider
                                    min={0}
                                    max={1}
                                    step={.01}
                                    className={styles.horizontalSlider}
                                    thumbClassName={styles.thumb}
                                    trackClassName={styles.track}
                                    renderThumb={(props, state) => <div {...props}>{state.valueNow}</div>}
                                />
                            </MenuItem>
                            <MenuItem>Change activation function
                                <ReactSlider
                                    min={0}
                                    max={1}
                                    step={.01}
                                    className={styles.horizontalSlider}
                                    thumbClassName={styles.thumb}
                                    trackClassName={styles.track}
                                    renderThumb={(props, state) => <div {...props}>{state.valueNow}</div>}
                                />
                            </MenuItem>

                        </SubMenu>
                    </SubMenu>
                </StyledMenu>
            </StyledProSidebar>
        </form>
    );
}
export default Settings;
