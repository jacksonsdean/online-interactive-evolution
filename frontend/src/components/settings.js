import React from "react";
import ReactSlider from 'react-slider'
import styles from "./Settings.module.css"
import styled from "styled-components";
import 'react-pro-sidebar/dist/css/styles.css';

import { ProSidebar, Menu, MenuItem, SubMenu } from 'react-pro-sidebar';

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
border-radius: 1%;
`;

const StyledDropDown = styled.select`
background-color: rgba(200,200,255,.1);
color:white;
margin-left:10px;
`;


function Settings() {
    return (
        <form>
            <StyledProSidebar>
                <StyledMenu iconShape="round">
                    <SubMenu title="Settings">
                        <SubMenu title="General">
                            <MenuItem>
                                Radial symmetry bias<input type="checkbox" />
                            </MenuItem>
                            <MenuItem>
                                {"Color mode"}
                                <StyledDropDown defaultValue="RGB">
                                    <option value="L">Grayscale</option>
                                    <option value="HSL">HSL</option>
                                    <option value="RGB">RGB</option>
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
