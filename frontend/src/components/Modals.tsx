import React, { useContext, useEffect, useRef, useState } from "react";
import AppContext, { MenuState } from "./interfaces";
import { colours, rgbaToHex } from "./interfaces";

import Toast from 'react-bootstrap/Toast'
import ToastContainer from "react-bootstrap/ToastContainer";
import Button from 'react-bootstrap/Button';
import ButtonGroup from "react-bootstrap/ButtonGroup";
import InputGroup from "react-bootstrap/InputGroup";
import Form from "react-bootstrap/Form";
import { getPhaseFraction } from "./imageLogic";
import { Table } from "react-bootstrap";


const _getCSSColour = (currentStateVal: any, targetStateVal: any, successPrefix: string, colourIdx: number): string => {
    // Boring function to map a success to current labelling colour. Used for GUI elements.
    const c = colours[colourIdx];
    const hex = rgbaToHex(c[0], c[1], c[2], 255);
    const matches: boolean = (currentStateVal === targetStateVal);

    const outlineStr = (matches) ? successPrefix + hex : 'white'
    return outlineStr;
}

const PhaseSelect = () => {
    const {
        imageInfo: [imageInfo,],
        selectedPhase: [selectedPhase, setSelectedPhase],
        menuState: [, setMenuState]
    } = useContext(AppContext)!

    const classes: number[] = Array.from(new Array(imageInfo!.nPhases), (_, i) => i + 1);

    const getStyle = (i: number) => {
        return {
            backgroundColor: _getCSSColour(i, selectedPhase, "", selectedPhase),
            border: _getCSSColour(i, i, "2px solid", i),
            margin: '1px 1px 1px 1px'
        }
    }

    const confirm = () => {
        if ((selectedPhase > 0) && (selectedPhase < 7)) {
            setMenuState('conf');
            return;
        } else {
            return;
        }
    }

    return (
        <>
            <span>Choose phase to analyze representativity of:</span>
            <ButtonGroup style={{ paddingLeft: "3%", marginLeft: '0%' }}>
                {
                    classes.map(i => <Button key={i} variant="light" onClick={(e) => setSelectedPhase(i)} style={getStyle(i)}>{i}</Button>)
                }
            </ButtonGroup>
            <Button variant="dark" onClick={(e) => { confirm() }}>Confirm</Button>
        </>
    );
}

const ConfidenceSelect = () => {
    const {
        imageInfo: [imageInfo,],
        selectedPhase: [selectedPhase,],
        menuState: [, setMenuState]
    } = useContext(AppContext)!
    const dimString = `${imageInfo?.nDims}D`;

    const phaseFrac = getPhaseFraction(
        imageInfo?.previewData.data!,
        imageInfo?.phaseVals[selectedPhase - 1]!
    ).toFixed(1);

    // make the stats a table
    // colour selected phase with correct colour
    // move confidence to bottom
    // add phase fraction for selected phase as a stat!

    return (
        <>
            <Table>
                <tbody>
                    <tr>
                        <td>Image Dimension:</td>
                        <td>{dimString}</td>
                    </tr>
                    <tr>
                        <td  >Chosen Phase:</td>
                        <td >{selectedPhase}</td>
                    </tr>
                    <tr>
                        <td>Volume Fraction (%):</td>
                        <td>{phaseFrac}</td>
                    </tr>
                    <tr>
                        <td>Estimated Time:</td>
                        <td>5s</td>
                    </tr>
                </tbody>
            </Table>
            <InputGroup>
                <InputGroup.Text>Confidence in Bounds (%):</InputGroup.Text>
                <Form.Control type="number" min={0} max={100} value={95} width={1} readOnly={true} size="sm"></Form.Control>
            </InputGroup>
            <Button variant="dark" onClick={(e) => { }}>Confirm</Button>
        </>
    )
}


const getMenuInfo = (state: MenuState) => {
    switch (state) {
        case 'phase':
            return { title: "Select Phase", innerHTML: <PhaseSelect /> }
        case 'conf':
            return { title: "Select Confidence", innerHTML: <ConfidenceSelect /> }
        case 'hidden': // fall through
        default:
            return { title: "", innerHTML: <></> }
    }
}

const Menu = () => {
    const {
        menuState: [menuState,]
    } = useContext(AppContext)!

    return (
        <>
            <ToastContainer className="p-5" position="bottom-end">
                <Toast show={menuState != 'hidden'}>
                    <Toast.Header className="roundedme-2">
                        <strong className="me-auto" style={{ fontSize: '1.5em' }}>{getMenuInfo(menuState).title}</strong>
                    </Toast.Header>
                    <Toast.Body>
                        {getMenuInfo(menuState).innerHTML}
                    </Toast.Body>
                </Toast>
            </ToastContainer>
        </>
    )
}

export default Menu;