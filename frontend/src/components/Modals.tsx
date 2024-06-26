import React, { useContext, useEffect, useRef, useState } from "react";
import AppContext, { MenuState } from "./interfaces";
import { colours, rgbaToHex } from "./interfaces";

import Toast from 'react-bootstrap/Toast'
import ToastContainer from "react-bootstrap/ToastContainer";
import Button from 'react-bootstrap/Button';
import ButtonGroup from "react-bootstrap/ButtonGroup";
import InputGroup from "react-bootstrap/InputGroup";
import Form from "react-bootstrap/Form";
import Spinner from "react-bootstrap/Spinner";
import Table from "react-bootstrap/Table";
import FormRange from 'react-bootstrap/FormRange';

import { getPhaseFraction } from "./imageLogic";


const centreStyle = { display: 'flex', justifyContent: 'center', alignItems: 'center', marginTop: '1em' }

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
            <p>Choose phase to analyze representativity of:</p>
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
        selectedConf: [selectedConf, setSelectedConf],
        errVF: [errVF, setErrVF],
        accurateFractions: [accurateFractions,],
        menuState: [, setMenuState]
    } = useContext(AppContext)!
    const dimString = `${imageInfo?.nDims}D`;

    const vals = imageInfo?.phaseVals!
    // horrible ternary: if server has responded and set the accurate phase fractions,
    // then use those values in the modal. If not, use the estimate from the first image
    const phaseFrac = (accurateFractions != null) ?
        accurateFractions[vals[selectedPhase - 1]].toFixed(3)
        : getPhaseFraction(
            imageInfo?.previewData.data!,
            vals[selectedPhase - 1]
        ).toFixed(3);

    const setConf = (e: any) => {
        setSelectedConf(Number(e.target!.value))
    }

    const setErr = (e: any) => {
        setErrVF(Number(e.target!.value))
    }

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
                        <td>Volume Fraction:</td>
                        <td>{phaseFrac}</td>
                    </tr>
                    <tr>
                        <td>Estimated Time:</td>
                        <td>5s</td>
                    </tr>
                </tbody>
            </Table>
            <InputGroup>
                <InputGroup.Text>Error Target (%):</InputGroup.Text>
                <Form.Control type="number" min={0} max={100} value={errVF} onChange={(e) => setErr(e)} width={1} size="sm"></Form.Control>
            </InputGroup>
            <InputGroup>
                <InputGroup.Text>Confidence in Bounds (%):</InputGroup.Text>
                <Form.Control type="number" min={0} max={100} value={selectedConf} onChange={(e) => setConf(e)} width={1} size="sm"></Form.Control>
            </InputGroup>
            <div style={centreStyle}>
                <Button variant="dark" onClick={(e) => { setMenuState('processing') }}>Calculate!</Button>
            </div>
        </>
    )
}

const Result = () => {
    // LB < VF_true < UB with '$CONF$% confidence  
    // need L = $N$pix for \epsilon = ...
    // epsilon slider (updates bounds in line 1)
    // conf re-select
    // CSS zoom anim on canvas
    const {
        analysisInfo: [analysisInfo,],
        imageInfo: [imageInfo,],
        selectedPhase: [selectedPhase,],
        selectedConf: [selectedConf, setSelectedConf],
        errVF: [errVF, setErrVF],
        accurateFractions: [accurateFractions,],
        menuState: [, setMenuState]
    } = useContext(AppContext)!

    // we have two errVFs here because we want the values in the text to reflect the old
    // errVF, the one they sent to the server and the slider to represent the new one
    // which they are setting for recalculate.
    const [newErrVF, setNewErrVF] = useState<number>(5);

    const vals = imageInfo?.phaseVals!
    const phaseFrac = (accurateFractions != null) ?
        accurateFractions[vals[selectedPhase - 1]]
        : getPhaseFraction(
            imageInfo?.previewData.data!,
            vals[selectedPhase - 1]
        );

    const perErr = analysisInfo?.percentageErr;
    const [LB, UB] = [Math.max((1 - perErr! / 100) * phaseFrac, 0), Math.min((1 + perErr! / 100) * phaseFrac, 1)];
    const [targLB, targUB] = [(1 - errVF! / 100) * phaseFrac, (1 + errVF! / 100) * phaseFrac];

    const l = analysisInfo?.lForDefaultErr;

    const setErr = (e: any) => {
        setNewErrVF(Number(e.target!.value))
    };

    const setConf = (e: any) => {
        setSelectedConf(Number(e.target!.value))
    };

    const recalculate = () => {
        setErrVF(newErrVF);
        setMenuState('processing');
    }

    return (
        <>
            <p>
                The bulk volume fraction ϕ is within {perErr}% of your image volume
                fraction Φ ({phaseFrac.toFixed(3)}) with {selectedConf}% confidence.
            </p>
            <p><b><i>i.e,</i> {LB.toFixed(3)} ≤ ϕ ≤ {UB.toFixed(3)} with {selectedConf}% confidence.</b></p>
            <p>For a {errVF.toFixed(2)}% error target, you need an image length of {l}px at the same resolution, which would give: </p>
            <p><b>&nbsp;&nbsp;&nbsp;&nbsp; {targLB.toFixed(3)} ≤ ϕ ≤ {targUB.toFixed(3)} with {selectedConf}% confidence.</b></p>
            <InputGroup>
                <InputGroup.Text>Error Target (%):</InputGroup.Text>
                <Form.Control type="number" min={0} max={100} value={newErrVF} onChange={(e) => setErr(e)} width={1} size="sm"></Form.Control>
            </InputGroup>
            <InputGroup>
                <InputGroup.Text>Confidence in Bounds (%):</InputGroup.Text>
                <Form.Control type="number" min={0} max={100} value={selectedConf} onChange={(e) => setConf(e)} width={1} size="sm"></Form.Control>
            </InputGroup>
            <div style={centreStyle}>
                <Button variant="dark" onClick={(e) => { recalculate() }}>Recalculate!</Button>
            </div>
        </>
    )
}


const getMenuInfo = (state: MenuState) => {
    switch (state) {
        case 'phase':
            return { title: "Select Phase", innerHTML: <PhaseSelect /> }
        case 'conf':
            return { title: "Select Confidence", innerHTML: <ConfidenceSelect /> }
        case 'processing':
            return { title: "Processing", innerHTML: <div style={centreStyle}><Spinner /></div> }
        case 'conf_result':
            return { title: "Result", innerHTML: <Result /> }
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
            <ToastContainer className="p-5" position="bottom-end" >
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